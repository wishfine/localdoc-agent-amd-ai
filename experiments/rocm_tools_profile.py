"""
Collect ROCm tool evidence and generate tuning notes for the report.

Outputs:
- results/rocm_tools_summary.csv
- results/rocm_tuning_recommendations.md
- results/amd_smi_list.txt
- results/amd_smi_static.txt
- results/amd_smi_metric.txt
- results/rocm_smi_performance.txt
- results/rocm_bandwidth_test.txt
- results/rocprofiler_tools.txt
- results/rocprofiler_run.txt

The script is intentionally non-fatal. Missing ROCm tools are recorded in the
CSV instead of failing the whole experiment run, so CPU/Jupyter development
environments still produce honest evidence.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

COMMON_ROCM_BIN_DIRS = [
    Path("/opt/rocm/bin"),
    Path("/usr/local/bin"),
    Path("/usr/bin"),
]


def _candidate_path(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for directory in COMMON_ROCM_BIN_DIRS:
        candidate = directory / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _resolve_any(names: Sequence[str]) -> tuple[Optional[str], str]:
    for name in names:
        resolved = _candidate_path(name)
        if resolved:
            return resolved, name
    return None, names[0]


def _resolve_python_executable(python_exe: str) -> str:
    """Return an executable Python path for child tools such as rocprofv3."""
    if python_exe:
        path = Path(python_exe)
        if path.is_absolute() and path.exists() and os.access(path, os.X_OK):
            return str(path)
        resolved = shutil.which(python_exe)
        if resolved:
            return resolved

    for candidate in (sys.executable, "python3", "python"):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_absolute() and path.exists() and os.access(path, os.X_OK):
            return str(path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return python_exe or "python3"


def _display_command(command: Sequence[str], display_name: str) -> str:
    parts = [display_name, *command[1:]]
    return " ".join(parts)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_command(
    names: Sequence[str],
    args: Sequence[str],
    output_path: Path,
    timeout_s: int,
) -> Dict[str, Any]:
    executable, display_name = _resolve_any(names)
    command_for_display = " ".join([display_name, *args])

    if executable is None:
        text = f"COMMAND NOT FOUND: {' or '.join(names)}\n"
        _write_text(output_path, text)
        return {
            "available": False,
            "exit_code": "",
            "command": command_for_display,
            "output_file": str(output_path),
            "duration_s": "",
            "note": text.strip(),
        }

    command = [executable, *args]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        duration = time.perf_counter() - started
        output = (proc.stdout or "") + (proc.stderr or "")
        if not output.strip():
            output = "(command produced no output)\n"
        header = (
            f"$ {_display_command(command, display_name)}\n"
            f"# exit_code={proc.returncode} duration_s={duration:.3f}\n\n"
        )
        _write_text(output_path, header + output)
        return {
            "available": True,
            "exit_code": proc.returncode,
            "command": _display_command(command, display_name),
            "output_file": str(output_path),
            "duration_s": round(duration, 3),
            "note": "ok" if proc.returncode == 0 else "command returned non-zero",
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        output = (exc.stdout or "") + (exc.stderr or "")
        text = (
            f"$ {command_for_display}\n"
            f"# TIMEOUT after {timeout_s}s duration_s={duration:.3f}\n\n"
            f"{output}\n"
        )
        _write_text(output_path, text)
        return {
            "available": True,
            "exit_code": "timeout",
            "command": command_for_display,
            "output_file": str(output_path),
            "duration_s": round(duration, 3),
            "note": f"timeout after {timeout_s}s",
        }
    except Exception as exc:  # pragma: no cover - defensive evidence path
        duration = time.perf_counter() - started
        text = (
            f"$ {command_for_display}\n"
            f"# FAILED duration_s={duration:.3f}\n\n"
            f"{type(exc).__name__}: {exc}\n"
        )
        _write_text(output_path, text)
        return {
            "available": True,
            "exit_code": -1,
            "command": command_for_display,
            "output_file": str(output_path),
            "duration_s": round(duration, 3),
            "note": f"{type(exc).__name__}: {exc}",
        }


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _record(
    rows: List[Dict[str, Any]],
    tool: str,
    category: str,
    used_for: str,
    grading_value: str,
    result: Dict[str, Any],
    note: str = "",
) -> None:
    rows.append(
        {
            "tool": tool,
            "category": category,
            "available": result.get("available", False),
            "exit_code": result.get("exit_code", ""),
            "command": result.get("command", ""),
            "output_file": result.get("output_file", ""),
            "duration_s": result.get("duration_s", ""),
            "used_for": used_for,
            "grading_value": grading_value,
            "note": note or result.get("note", ""),
        }
    )


def _small_rocm_probe_script(results_dir: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="_rocprofv3_probe.py",
        delete=False,
    )
    probe_path = Path(handle.name)
    probe_code = """
import time

try:
    import torch
except Exception as exc:
    print(f"torch_import_failed={type(exc).__name__}: {exc}")
    raise SystemExit(0)

print("torch_version=", torch.__version__)
print("torch_hip_version=", getattr(torch.version, "hip", None))
print("cuda_available=", torch.cuda.is_available())
if not (getattr(torch.version, "hip", None) and torch.cuda.is_available()):
    raise SystemExit(0)

device = "cuda"
torch.manual_seed(2026)
a = torch.randn((512, 512), device=device, dtype=torch.float16)
b = torch.randn((512, 512), device=device, dtype=torch.float16)
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
print("probe_avg_ms=", (time.perf_counter() - t0) * 1000 / 5)
print("probe_output_norm=", float(c.float().norm().detach().cpu()))
"""
    handle.write(textwrap.dedent(probe_code).lstrip())
    handle.close()
    return probe_path


def _build_rocprofv3_probe_command(
    executable: str,
    python_exe: str,
    probe_script: Path,
    profiler_dir: Path,
) -> List[str]:
    return [
        executable,
        "--runtime-trace",
        "--kernel-trace",
        "--memory-copy-trace",
        "--stats",
        "--summary",
        "--output-format",
        "csv",
        "--output-directory",
        str(profiler_dir),
        "--",
        python_exe,
        str(probe_script),
    ]


def _collect_profiler_help(results_dir: Path) -> Dict[str, Any]:
    output_path = results_dir / "rocprofiler_tools.txt"
    sections: List[str] = []
    available_any = False
    exit_codes: List[str] = []

    profiler_commands = [
        ("rocprofv3", ["rocprofv3"], ["--version"]),
        ("rocprofv3_help", ["rocprofv3"], ["--help"]),
        ("rocprof_compute", ["rocprof-compute"], ["--help"]),
        ("rocprof_sys", ["rocprof-sys", "rocsys"], ["--help"]),
        ("rocprof_legacy", ["rocprof"], ["--help"]),
        ("rocprofv2_legacy", ["rocprofv2"], ["--help"]),
    ]

    for label, names, args in profiler_commands:
        executable, display_name = _resolve_any(names)
        sections.append(f"\n===== {label}: {' or '.join(names)} {' '.join(args)} =====\n")
        if executable is None:
            sections.append(f"COMMAND NOT FOUND: {' or '.join(names)}\n")
            exit_codes.append(f"{label}:missing")
            continue
        available_any = True
        try:
            proc = subprocess.run(
                [executable, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            sections.append(f"$ {display_name} {' '.join(args)}\n")
            sections.append(f"# exit_code={proc.returncode}\n")
            sections.append((proc.stdout or "") + (proc.stderr or "") or "(no output)\n")
            exit_codes.append(f"{label}:{proc.returncode}")
        except Exception as exc:
            sections.append(f"FAILED: {type(exc).__name__}: {exc}\n")
            exit_codes.append(f"{label}:-1")

    _write_text(output_path, "".join(sections).lstrip())
    return {
        "available": available_any,
        "exit_code": ";".join(exit_codes),
        "command": "rocprofv3/rocprof-compute/rocprof-sys help and version probes",
        "output_file": str(output_path),
        "duration_s": "",
        "note": "profiler tools probed",
    }


def _run_rocprofv3_probe(results_dir: Path, python_exe: str, timeout_s: int) -> Dict[str, Any]:
    output_path = results_dir / "rocprofiler_run.txt"
    executable, display_name = _resolve_any(["rocprofv3"])
    if executable is None:
        text = "COMMAND NOT FOUND: rocprofv3\n"
        _write_text(output_path, text)
        return {
            "available": False,
            "exit_code": "",
            "command": "rocprofv3 --runtime-trace --kernel-trace ...",
            "output_file": str(output_path),
            "duration_s": "",
            "note": "rocprofv3 not found",
        }

    probe_script = _small_rocm_probe_script(results_dir)
    profiler_dir = results_dir / "rocprofv3_probe"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    child_python = _resolve_python_executable(python_exe)
    command = _build_rocprofv3_probe_command(
        executable=executable,
        python_exe=child_python,
        probe_script=probe_script,
        profiler_dir=profiler_dir,
    )
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        duration = time.perf_counter() - started
        output = (proc.stdout or "") + (proc.stderr or "")
        generated = "\n".join(
            str(path.relative_to(results_dir))
            for path in sorted(profiler_dir.rglob("*"))
            if path.is_file()
        )
        text = (
            f"$ {_display_command(command, display_name)}\n"
            f"# exit_code={proc.returncode} duration_s={duration:.3f}\n"
            f"# generated_files_under_results:\n{generated or '(none)'}\n\n"
            f"{output or '(no output)'}\n"
        )
        _write_text(output_path, text)
        return {
            "available": True,
            "exit_code": proc.returncode,
            "command": _display_command(command, display_name),
            "output_file": str(output_path),
            "duration_s": round(duration, 3),
            "note": "rocprofv3 probe completed" if proc.returncode == 0 else "rocprofv3 probe returned non-zero",
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        text = (
            f"$ {' '.join(command)}\n"
            f"# TIMEOUT after {timeout_s}s duration_s={duration:.3f}\n\n"
            f"{(exc.stdout or '') + (exc.stderr or '')}\n"
        )
        _write_text(output_path, text)
        return {
            "available": True,
            "exit_code": "timeout",
            "command": _display_command(command, display_name),
            "output_file": str(output_path),
            "duration_s": round(duration, 3),
            "note": f"rocprofv3 probe timeout after {timeout_s}s",
        }
    finally:
        try:
            probe_script.unlink()
        except FileNotFoundError:
            pass


def _make_tuning_notes(rows: List[Dict[str, Any]], path: Path) -> None:
    available_tools = [row["tool"] for row in rows if str(row.get("available")) == "True"]
    unavailable_tools = [row["tool"] for row in rows if str(row.get("available")) != "True"]
    nonzero = [
        row["tool"]
        for row in rows
        if str(row.get("available")) == "True" and str(row.get("exit_code")) not in ("", "0")
    ]

    text = f"""# ROCm 工具性能检测与调优记录

生成时间：{datetime.now().isoformat(timespec="seconds")}

## 工具覆盖情况

- 已检测到的工具：{", ".join(available_tools) if available_tools else "无"}
- 未检测到或未安装的工具：{", ".join(unavailable_tools) if unavailable_tools else "无"}
- 命令非零退出或超时：{", ".join(nonzero) if nonzero else "无"}

## 本项目如何使用 ROCm 工具

| 工具 | 本项目用途 | 产物 |
|------|------------|------|
| rocminfo | 枚举 ROCm GPU agent、gfx 架构、ROCm 栈是否工作 | `results/rocminfo.txt` |
| AMD SMI | GPU 型号、驱动、温度、功耗、显存和利用率监控 | `results/amd_smi_*.txt` |
| ROCm SMI | 兼容旧环境的 GPU 功耗、显存、频率、温度、利用率监控 | `results/rocm_smi_performance.txt` |
| ROCm Bandwidth Test | 测 CPU-GPU/GPU-GPU 数据传输带宽，判断数据搬运瓶颈 | `results/rocm_bandwidth_test.txt` |
| rocprofv3 / ROCProfiler | 采集 HIP runtime、kernel、memory copy trace，定位 kernel 与拷贝瓶颈 | `results/rocprofiler_tools.txt`、`results/rocprofiler_run.txt` |
| ROCm Compute/System Profiler | 用于进一步做 CU/L2/Speed-of-Light 分析和 CPU/GPU 系统级 trace | `results/rocprofiler_tools.txt` |

## 调优结论写法

1. 如果 `results/matmul_benchmark.csv` 中 `ROCm_GPU` 相比 CPU 有明显 speedup，报告中可说明矩阵密集型任务适合放到 GPU。
2. 如果 `results/precision_compare.csv` 中 FP16 速度更快且 `relative_l2_error` 可接受，报告中可说明半精度能提升吞吐，但需要结合误差约束使用。
3. 如果 `results/rocm_bandwidth_test.txt` 显示 CPU-GPU 带宽较低，报告中要把数据搬运列为瓶颈，并说明应减少 host/device 往返、批量化传输、复用 GPU resident tensor。
4. 如果 `results/amd_smi_metric.txt` 或 `results/rocm_smi_performance.txt` 中 GPU 利用率低，报告中可说明当前 RAG/LLM 小批量负载未充分填满 GPU，应通过 batch、增大矩阵规模或并发请求提升占用率。
5. 如果 `results/power_trace.csv` 有 GPU power 采样，报告中用 `energy_summary.csv` 说明性能-能耗折中；如果没有采样，明确写成工具不可用或容器权限不足。
6. 如果 `results/rocprofiler_run.txt` 产生 trace/summary 文件，报告中截取 kernel/runtime/memory copy 摘要，用它支撑“调优依据来自 ROCm profiler”。

## 面向本项目的优化建议

- 基础实验：扩大 matmul size 和 batch size，让 ROCm GPU 相比 CPU 的优势更明显。
- LLM 推理：限制 `max_new_tokens`，固定 prompt 长度，分别报告 prefill/generation 或总 tokens/s。
- RAG 管线：embedding/query 保持批处理，减少逐 chunk Python 循环；文档入库先统一切块再统一 fit/embed。
- 数据搬运：PyTorch tensor 创建后尽量保留在 GPU，避免每次 query 都在 CPU/GPU 间复制。
- 能效：同时报告 latency、tokens/s、GPU power、estimated energy，不只报告耗时。

## 截图建议

- 打开 `results/rocm_tools_summary.csv`，截工具 available、command、output_file。
- 打开 `results/amd_smi_metric.txt` 或 `results/rocm_smi_performance.txt`，截 GPU power/temp/utilization。
- 打开 `results/rocm_bandwidth_test.txt`，截 CPU-GPU bandwidth。
- 打开 `results/rocprofiler_run.txt`，截 rocprofv3 命令和生成文件列表。
"""
    _write_text(path, text)


def _collect_tools(results_dir: Path, python_exe: str, profiler_timeout_s: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    _record(
        rows,
        "rocminfo",
        "system_management",
        "Report ROCm system information and enumerate GPU agents",
        "环境确认：GPU 名称、gfx 架构、ROCm 栈",
        _run_command(["rocminfo"], [], results_dir / "rocminfo.txt", timeout_s=30),
    )
    _record(
        rows,
        "amd-smi list",
        "system_management",
        "List AMD GPUs visible to AMD SMI",
        "环境确认：AMD GPU 是否可见",
        _run_command(["amd-smi"], ["list"], results_dir / "amd_smi_list.txt", timeout_s=20),
    )
    _record(
        rows,
        "amd-smi static",
        "system_management",
        "Collect static GPU and driver properties",
        "环境确认：GPU 型号、驱动、固件/硬件信息",
        _run_command(["amd-smi"], ["static"], results_dir / "amd_smi_static.txt", timeout_s=20),
    )
    _record(
        rows,
        "amd-smi metric",
        "system_management",
        "Collect live GPU utilization, power, temperature, and memory metrics",
        "能效分析：功耗、温度、利用率、显存",
        _run_command(["amd-smi"], ["metric"], results_dir / "amd_smi_metric.txt", timeout_s=20),
    )
    _record(
        rows,
        "rocm-smi",
        "system_management",
        "Collect ROCm SMI power, clocks, VRAM, temperature, and utilization",
        "能效分析：兼容旧 ROCm 环境的 GPU 资源/功耗证据",
        _run_command(
            ["rocm-smi"],
            [
                "--showproductname",
                "--showpower",
                "--showmeminfo",
                "vram",
                "--showclocks",
                "--showuse",
                "--showtemp",
            ],
            results_dir / "rocm_smi_performance.txt",
            timeout_s=20,
        ),
    )
    _record(
        rows,
        "rocm-bandwidth-test",
        "performance",
        "Measure CPU-GPU and GPU-GPU transfer bandwidth",
        "性能检测：数据搬运带宽和统一内存/主机设备传输瓶颈",
        _run_command(
            ["rocm-bandwidth-test", "rocm_bandwidth_test"],
            [],
            results_dir / "rocm_bandwidth_test.txt",
            timeout_s=90,
        ),
    )
    _record(
        rows,
        "rocprofiler tools",
        "performance",
        "Probe rocprofv3, rocprof-compute, rocprof-sys, and legacy profiler CLIs",
        "调优工具链：确认 profiler 是否可用",
        _collect_profiler_help(results_dir),
    )
    _record(
        rows,
        "rocprofv3 probe",
        "performance",
        "Run a small ROCm PyTorch matmul under rocprofv3 when available",
        "调优证据：HIP runtime、kernel、memory copy trace",
        _run_rocprofv3_probe(results_dir, python_exe, profiler_timeout_s),
    )

    return rows


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ROCm tool evidence and tuning notes")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--profiler-timeout", type=int, default=120)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ROCm tools profiling evidence")
    print("=" * 60)

    rows = _collect_tools(results_dir, args.python, args.profiler_timeout)
    summary_path = results_dir / "rocm_tools_summary.csv"
    notes_path = results_dir / "rocm_tuning_recommendations.md"
    _write_csv(rows, summary_path)
    _make_tuning_notes(rows, notes_path)

    print(f"  [saved] {summary_path}")
    print(f"  [saved] {notes_path}")
    for row in rows:
        exit_code = str(row.get("exit_code", ""))
        if not row["available"]:
            status = "MISS"
        elif exit_code in ("", "0"):
            status = "OK"
        else:
            status = "FAIL"
        print(f"  [{status}] {row['tool']}: {row['output_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
