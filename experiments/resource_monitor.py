"""
Resource and power monitor for benchmark runs.

Outputs:
- results/power_trace.csv
- results/energy_summary.csv

When ``rocm-smi`` is available, GPU power readings are sampled and integrated
into an estimated energy value. On machines without ROCm, the script still
records CPU and memory usage and clearly marks GPU power as unavailable.
"""

import argparse
import csv
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _psutil_stats() -> Dict[str, Any]:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": mem.percent,
            "memory_used_mb": round(mem.used / (1024 * 1024), 2),
        }
    except Exception:
        return {
            "cpu_percent": "",
            "memory_percent": "",
            "memory_used_mb": "",
        }


def _read_rocm_power() -> tuple[Optional[float], str]:
    if shutil.which("rocm-smi") is None:
        return None, "rocm-smi not found"
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showpower"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return None, f"rocm-smi failed: {type(exc).__name__}: {exc}"

    output = (proc.stdout or "") + (proc.stderr or "")
    watts = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*W", output)]
    if not watts:
        return None, output.strip().splitlines()[0] if output.strip() else "no watt reading"
    return sum(watts) / len(watts), "rocm-smi --showpower"


def sample(elapsed_s: float) -> Dict[str, Any]:
    stats = _psutil_stats()
    gpu_power_w, source = _read_rocm_power()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed_s, 3),
        **stats,
        "gpu_power_w": round(gpu_power_w, 3) if gpu_power_w is not None else "",
        "gpu_power_source": source,
        "is_rocm_power_available": gpu_power_w is not None,
    }


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
        writer.writerows([{k: row.get(k, "") for k in keys} for row in rows])


def summarize(rows: List[Dict[str, Any]], interval_s: float) -> Dict[str, Any]:
    powers = [float(r["gpu_power_w"]) for r in rows if r.get("gpu_power_w") not in ("", None)]
    elapsed = float(rows[-1]["elapsed_s"]) if rows else 0.0
    summary = {
        "samples": len(rows),
        "duration_s": round(elapsed, 3),
        "interval_s": interval_s,
        "rocm_power_samples": len(powers),
        "avg_gpu_power_w": "",
        "max_gpu_power_w": "",
        "estimated_gpu_energy_j": "",
        "note": "GPU power unavailable; install/use rocm-smi on AMD ROCm hardware.",
    }
    if powers:
        avg_power = sum(powers) / len(powers)
        summary.update({
            "avg_gpu_power_w": round(avg_power, 3),
            "max_gpu_power_w": round(max(powers), 3),
            "estimated_gpu_energy_j": round(avg_power * elapsed, 3),
            "note": "Energy estimated as average sampled GPU power multiplied by monitor duration.",
        })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample CPU/memory and ROCm GPU power during benchmarks")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Fixed duration. 0 means wait for stop file.")
    parser.add_argument("--stop-file", type=str, default=None)
    parser.add_argument("--max-duration", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    stop_file = Path(args.stop_file) if args.stop_file else None
    rows: List[Dict[str, Any]] = []
    start = time.perf_counter()

    print("=" * 60)
    print("  Resource / Power Monitor")
    print("=" * 60)
    try:
        while True:
            elapsed = time.perf_counter() - start
            rows.append(sample(elapsed))
            if args.duration > 0 and elapsed >= args.duration:
                break
            if stop_file and stop_file.exists():
                break
            if elapsed >= args.max_duration:
                break
            time.sleep(max(args.interval, 0.1))
    finally:
        trace_path = results_dir / "power_trace.csv"
        summary_path = results_dir / "energy_summary.csv"
        _write_csv(rows, trace_path)
        _write_csv([summarize(rows, args.interval)], summary_path)
        print(f"  [saved] {trace_path}")
        print(f"  [saved] {summary_path}")


if __name__ == "__main__":
    main()
