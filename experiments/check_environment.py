"""
环境检查脚本 - Environment Check

运行后检测当前硬件和软件环境，判断是否有真实的
AMD GPU (ROCm) 或 NPU (Ryzen AI SDK) 可用。

输出: results/environment_report.txt

运行方式:
    python experiments/check_environment.py
"""

import os
import sys
import platform
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def check_python() -> dict:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def check_memory() -> dict:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_memory_mb": round(mem.total / (1024 * 1024)),
            "available_memory_mb": round(mem.available / (1024 * 1024)),
            "memory_percent": mem.percent,
            "cpu_count": psutil.cpu_count(),
        }
    except ImportError:
        return {"note": "psutil not installed, memory info unavailable"}


def check_torch() -> dict:
    result = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_version": None,
        "hip_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_arch": None,
    }
    try:
        import torch
        result["torch_installed"] = True
        result["torch_version"] = torch.__version__
        result["cuda_version"] = getattr(torch.version, "cuda", None)
        result["hip_version"] = getattr(torch.version, "hip", None)
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_count"] = torch.cuda.device_count()
            try:
                result["gpu_arch"] = torch.cuda.get_device_capability(0)
            except Exception:
                pass
    except ImportError:
        pass
    return result


def check_kernel() -> dict:
    """Check kernel version (important for Strix Halo / Ryzen AI MAX+).

    Requirements from ROCm docs:
    - Ubuntu 24.04 HWE: >= 6.17.0-19.19~24.04.2
    - Ubuntu 24.04 OEM: >= 6.14.0-1018
    - Other distros: >= 6.18.4
    Only applies to Linux. macOS/Windows skip the check.
    """
    result = {
        "kernel_version": platform.release(),
        "is_strix_halo_compatible": None,
        "min_kernel_note": "",
    }

    # Only check on Linux
    if platform.system() != "Linux":
        result["is_strix_halo_compatible"] = None
        result["min_kernel_note"] = (
            f"Platform is {platform.system()}, not Linux. "
            "Strix Halo kernel check skipped."
        )
        return result

    try:
        kernel = platform.release()
        parts = kernel.split(".")
        if len(parts) >= 2:
            major, minor = int(parts[0]), int(parts[1])
            if major > 6 or (major == 6 and minor >= 18):
                result["is_strix_halo_compatible"] = True
            elif major == 6 and minor >= 14:
                result["is_strix_halo_compatible"] = True
                result["min_kernel_note"] = (
                    f"Kernel {kernel} >= 6.14: may work on Ubuntu OEM (>= 6.14.0-1018), "
                    "but >= 6.17.0 (HWE) or >= 6.18.4 (other) recommended for stability."
                )
            else:
                result["is_strix_halo_compatible"] = False
                result["min_kernel_note"] = (
                    f"Kernel {kernel} too old for Strix Halo. "
                    "Need >= 6.17.0 (Ubuntu HWE) or >= 6.18.4 (other distros)."
                )
    except (ValueError, IndexError):
        pass
    return result


def check_onnxruntime() -> dict:
    result = {
        "onnxruntime_installed": False,
        "onnxruntime_version": None,
        "available_providers": [],
        "vitisai_available": False,
        "ryzenai_available": False,
        "directml_available": False,
    }
    try:
        import onnxruntime as ort
        result["onnxruntime_installed"] = True
        result["onnxruntime_version"] = ort.__version__
        providers = ort.get_available_providers()
        result["available_providers"] = providers
        result["vitisai_available"] = "VitisAIExecutionProvider" in providers
        result["ryzenai_available"] = "RyzenAIExecutionProvider" in providers
        result["directml_available"] = "DmlExecutionProvider" in providers
    except ImportError:
        pass
    return result


def determine_conclusion(torch_info: dict, ort_info: dict, kernel_info: dict) -> dict:
    rocm_available = (
        torch_info.get("hip_version") is not None
        and torch_info.get("cuda_available", False)
    )
    npu_available = (
        ort_info.get("vitisai_available", False)
        or ort_info.get("ryzenai_available", False)
        or ort_info.get("directml_available", False)
    )

    if rocm_available:
        gpu_mode = "real ROCm GPU"
    elif torch_info.get("cuda_available"):
        gpu_mode = "CUDA GPU (not AMD ROCm)"
    else:
        gpu_mode = "unavailable"

    if npu_available:
        npu_mode = "real NPU provider detected"
    else:
        npu_mode = "unavailable"

    if rocm_available or npu_available:
        current_mode = "real hardware"
    else:
        current_mode = "CPU fallback + simulated backend"

    return {
        "rocm_gpu_available": rocm_available,
        "amd_npu_available": npu_available,
        "gpu_mode": gpu_mode,
        "npu_mode": npu_mode,
        "current_mode": current_mode,
        "kernel_version": kernel_info.get("kernel_version", "unknown"),
        "is_strix_halo_compatible": kernel_info.get("is_strix_halo_compatible"),
        "min_kernel_note": kernel_info.get("min_kernel_note", ""),
    }


def main():
    print("=" * 60)
    print("  环境检查 (Environment Check)")
    print("=" * 60)

    py_info = check_python()
    mem_info = check_memory()
    torch_info = check_torch()
    ort_info = check_onnxruntime()
    kernel_info = check_kernel()
    conclusion = determine_conclusion(torch_info, ort_info, kernel_info)

    # Build report
    lines = [
        "=" * 60,
        "  LocalDoc Agent - 环境检查报告",
        f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "## Python 环境",
        f"  Python 版本: {py_info['python_version']}",
        f"  可执行文件: {py_info['python_executable']}",
        f"  平台: {py_info['platform']}",
        f"  架构: {py_info['machine']}",
        f"  处理器: {py_info['processor']}",
        "",
        "## 内核信息",
        f"  内核版本: {kernel_info['kernel_version']}",
        f"  Strix Halo 兼容: {kernel_info['is_strix_halo_compatible']}",
    ]
    if kernel_info["min_kernel_note"]:
        lines.append(f"  ⚠️ {kernel_info['min_kernel_note']}")

    lines += ["", "## 内存信息"]
    for k, v in mem_info.items():
        lines.append(f"  {k}: {v}")

    lines += [
        "",
        "## PyTorch 环境",
        f"  torch 已安装: {torch_info['torch_installed']}",
        f"  torch 版本: {torch_info['torch_version']}",
        f"  torch.version.cuda: {torch_info['cuda_version']}",
        f"  torch.version.hip: {torch_info['hip_version']}",
        f"  torch.cuda.is_available(): {torch_info['cuda_available']}",
        f"  GPU 名称: {torch_info['gpu_name']}",
        f"  GPU 数量: {torch_info['gpu_count']}",
        "",
        "## ONNX Runtime 环境",
        f"  onnxruntime 已安装: {ort_info['onnxruntime_installed']}",
        f"  onnxruntime 版本: {ort_info['onnxruntime_version']}",
        f"  可用 EP 列表: {ort_info['available_providers']}",
        f"  VitisAIExecutionProvider: {ort_info['vitisai_available']}",
        f"  DmlExecutionProvider: {ort_info['directml_available']}",
        f"  RyzenAIExecutionProvider: {ort_info.get('ryzenai_available', False)}",
        "",
        "## 结论",
        f"  ROCm GPU available: {conclusion['rocm_gpu_available']}",
        f"  AMD NPU available: {conclusion['amd_npu_available']}",
        f"  GPU mode: {conclusion['gpu_mode']}",
        f"  NPU mode: {conclusion['npu_mode']}",
        f"  Current mode: {conclusion['current_mode']}",
        "",
        "## 注意事项",
        "  - ROCm 覆盖 GPU 计算，不覆盖 NPU",
        "  - Ryzen APU 上 ROCm 仅支持 PyTorch（不支持 TF/JAX/ONNX）",
        "  - NPU 推理需通过 Ryzen AI SDK 或 Lemonade",
        "  - Strix Halo 需要 kernel >= 6.17.0 (HWE) 或 >= 6.18.4",
        "",
    ]

    if conclusion["current_mode"] == "CPU fallback + simulated backend":
        lines.append(
            "⚠️ 当前没有检测到真实 AMD GPU/NPU 硬件。"
            "所有 GPU/NPU 相关实验数据将为 simulated backend 结果。"
        )

    report = "\n".join(lines)

    # Print to console
    print(report)

    # Save to file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "environment_report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()
