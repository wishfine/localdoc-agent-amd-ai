"""
Plot course-rubric basic benchmark results.

Inputs:
- results/matmul_benchmark.csv
- results/precision_compare.csv
- results/mlp_train_log.csv

Outputs:
- figures/matmul_benchmark.png
- figures/precision_compare.png
- figures/mlp_training_curve.png
- figures/energy_comparison.png
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

BACKEND_COLORS = {
    "CPU": "#4e79a7",
    "Torch_CPU": "#59a14f",
    "ROCm_GPU": "#f28e2b",
}


def _setup_plot_style() -> None:
    """Use ASCII-only labels so headless Linux containers do not need CJK fonts."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _convert(value: str) -> Any:
    if value == "":
        return value
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_csv(filename: str) -> List[Dict[str, Any]]:
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  [skip] Missing file: {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        return [{k: _convert(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _available_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("available") is True and r.get("measurement_type") != "unavailable"]


def plot_matmul(output_path: Optional[Path] = None) -> Optional[Path]:
    rows = _available_rows(_read_csv("matmul_benchmark.csv"))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    fig.suptitle(
        "Matrix Multiplication Benchmark (FP32, log scale)",
        fontsize=15,
        fontweight="bold",
    )
    for backend in sorted({r["backend"] for r in rows}):
        data = sorted([r for r in rows if r["backend"] == backend], key=lambda r: r["size"])
        ax.plot(
            [r["size"] for r in data],
            [r["avg_ms"] for r in data],
            "o-",
            label=backend,
            linewidth=2,
            color=BACKEND_COLORS.get(backend),
        )
    ax.set_xlabel("Matrix size N (N x N)")
    ax.set_ylabel("Average latency (ms)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.grid(True, which="minor", axis="y", alpha=0.15)
    ax.legend()
    out = output_path or (FIGURES_DIR / "matmul_benchmark.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


def plot_precision(output_path: Optional[Path] = None) -> Optional[Path]:
    rows = _available_rows(_read_csv("precision_compare.csv"))
    if not rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(
        "FP32 / FP16 Performance and Error (latency uses log scale)",
        fontsize=15,
        fontweight="bold",
    )

    labels = [f"{r['backend']}\nN={r['size']}" for r in rows]
    x = list(range(len(rows)))
    width = 0.38
    axes[0].bar([i - width / 2 for i in x], [r["fp32_ms"] for r in rows], width, label="FP32", color="#4e79a7")
    axes[0].bar([i + width / 2 for i in x], [r["fp16_ms"] for r in rows], width, label="FP16", color="#f28e2b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Average latency (ms)")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].grid(axis="y", which="minor", alpha=0.15)
    axes[0].legend()

    axes[1].bar(labels, [r["mean_abs_error"] for r in rows], color="#e15759")
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", labelrotation=20)

    out = output_path or (FIGURES_DIR / "precision_compare.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


def plot_mlp(output_path: Optional[Path] = None) -> Optional[Path]:
    rows = _available_rows(_read_csv("mlp_train_log.csv"))
    rows = [r for r in rows if r.get("epoch") != ""]
    if not rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Single-device MLP Training Curve", fontsize=15, fontweight="bold")

    for backend in sorted({r["backend"] for r in rows}):
        data = sorted([r for r in rows if r["backend"] == backend], key=lambda r: r["epoch"])
        color = BACKEND_COLORS.get(backend)
        axes[0].plot([r["epoch"] for r in data], [r["loss"] for r in data], "o-", label=backend, color=color)
        axes[1].plot([r["epoch"] for r in data], [r["accuracy"] for r in data], "o-", label=backend, color=color)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    out = output_path or (FIGURES_DIR / "mlp_training_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


def plot_energy(output_path: Optional[Path] = None) -> Optional[Path]:
    rows = _read_csv("power_trace.csv")
    if not rows:
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle("Resource and Energy Sampling", fontsize=15, fontweight="bold")
    x = [r.get("elapsed_s", 0) for r in rows]
    cpu = [r.get("cpu_percent", 0) if r.get("cpu_percent", "") != "" else 0 for r in rows]
    mem = [r.get("memory_percent", 0) if r.get("memory_percent", "") != "" else 0 for r in rows]
    ax1.plot(x, cpu, "o-", label="CPU utilization (%)", color="#4e79a7")
    ax1.plot(x, mem, "o-", label="Memory utilization (%)", color="#76b7b2")
    ax1.set_xlabel("Elapsed (s)")
    ax1.set_ylabel("CPU / Memory (%)")
    ax1.grid(True, alpha=0.3)

    power_values = [r.get("gpu_power_w", "") for r in rows]
    power = [float(v) for v in power_values if v != ""]
    if power:
        ax2 = ax1.twinx()
        power_x = [r.get("elapsed_s", 0) for r in rows if r.get("gpu_power_w", "") != ""]
        ax2.plot(power_x, power, "s-", label="GPU Power (W)", color="#e15759")
        ax2.set_ylabel("GPU Power (W)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.text(
            0.5, 0.12,
            "No GPU power samples in current CSV: use as CPU/memory trace only",
            transform=ax1.transAxes,
            ha="center",
            fontsize=10,
        )
        ax1.legend(loc="best")

    out = output_path or (FIGURES_DIR / "energy_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot basic heterogeneous benchmark results")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--figures-dir", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    global RESULTS_DIR, FIGURES_DIR
    args = parse_args(argv)
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
    if args.figures_dir:
        FIGURES_DIR = Path(args.figures_dir)

    print("=" * 60)
    print("  Basic Benchmark Plotting")
    print("=" * 60)
    _setup_plot_style()
    plot_matmul()
    plot_precision()
    plot_mlp()
    plot_energy()
    print("=" * 60)


if __name__ == "__main__":
    main()
