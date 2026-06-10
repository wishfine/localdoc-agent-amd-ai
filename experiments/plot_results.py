"""
Plot Benchmark Results

- figures/latency_comparison.png
- figures/backend_comparison.png
- figures/resource_usage.png
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def _setup_plot_style() -> None:
    """Use ASCII-only labels so headless Linux containers do not need CJK fonts."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _read_csv(filename: str) -> List[Dict[str, Any]]:
    """Read a CSV file from RESULTS_DIR and return list of dicts."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"  [error] Missing file: {filepath}")
        return []
    rows: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Attempt numeric conversion
            typed: Dict[str, Any] = {}
            for k, v in row.items():
                try:
                    typed[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        typed[k] = float(v)
                    except (ValueError, TypeError):
                        typed[k] = v
            rows.append(typed)
    return rows


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

BACKEND_COLORS = {
    "CPU": "#4e79a7",
    "GPU": "#f28e2b",
    "NPU": "#e15759",
    "SimulatedNPU": "#76b7b2",
}

BACKEND_LABELS = {
    "CPU": "CPU",
    "GPU": "GPU",
    "NPU": "NPU",
    "SimulatedNPU": "Simulated NPU (simulated)",
}


def plot_latency_comparison(
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate latency comparison chart from latency_results.csv.

    Returns the path to the saved figure, or None if data is missing.
    """
    rows = _read_csv("latency_results.csv")
    if not rows:
        print("  [skip] latency_results.csv is empty or missing")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(
        "Heterogeneous Scheduling Latency Benchmark\n"
        "measurement_type distinguishes real_hardware / simulated / unavailable",
        fontsize=14, fontweight="bold",
    )

    # Support both old (benchmark_latency) and new (benchmark_real) test names
    test_types = [
        ("embedding", "Document Ingest / Embedding", "Document count", "doc_count"),
        ("query_embedding", "Query Embedding", "Chunk count", "chunk_count"),
        ("end_to_end_rag", "End-to-end RAG", "Document count", "doc_count"),
    ]

    for ax, (test_key, title, xlabel, xkey) in zip(axes, test_types):
        subset = [r for r in rows if r.get("test") == test_key]
        # Filter out unavailable backends
        subset = [r for r in subset if r.get("measurement_type") != "unavailable"]
        if not subset:
            ax.set_title(f"{title} (no data)")
            continue

        backends_in_data = sorted({r["backend"] for r in subset})
        for backend in backends_in_data:
            data = sorted(
                [r for r in subset if r["backend"] == backend],
                key=lambda r: r.get(xkey, 0),
            )
            x = [r.get(xkey, 0) for r in data]
            y = [r["latency_ms"] for r in data]
            is_simulated = any(str(r.get("is_simulated", "")).lower() == "true" for r in data)
            ax.plot(
                x, y,
                "o--" if is_simulated else "o-",
                label=BACKEND_LABELS.get(backend, backend),
                color=BACKEND_COLORS.get(backend, None),
                linewidth=2,
                markersize=6,
                alpha=0.75 if is_simulated else 1.0,
            )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.grid(True, which="minor", axis="y", alpha=0.15)

    out = output_path or (FIGURES_DIR / "latency_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


def plot_backend_comparison(
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate backend performance bar chart from backend_results.csv."""
    rows = _read_csv("backend_results.csv")
    if not rows:
        print("  [skip] backend_results.csv is empty or missing")
        return None

    # Keep the main chart to real hardware so simulated NPU fallback does not
    # visually dominate the CPU/GPU comparison used in the report.
    excluded = [
        r for r in rows
        if r.get("measurement_type") == "unavailable"
        or str(r.get("is_simulated", "")).lower() == "true"
    ]
    rows = [
        r for r in rows
        if r.get("measurement_type") != "unavailable"
        and str(r.get("is_simulated", "")).lower() != "true"
    ]
    if not rows:
        print("  [skip] backend_results.csv has no real hardware backend rows")
        return None

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.suptitle(
        "Real Backend Performance Comparison - Average Latency\n"
        "(simulated/unavailable backends excluded from bars)",
        fontsize=14, fontweight="bold",
    )

    backends = [r["backend"] for r in rows]
    avg_ms = [r["avg_latency_ms"] for r in rows]
    mtypes = [r.get("measurement_type", "unknown") for r in rows]
    colors = [BACKEND_COLORS.get(b, "#999999") for b in backends]
    labels = [f"{BACKEND_LABELS.get(b, b)}\n[{mt}]" for b, mt in zip(backends, mtypes)]

    bars = ax.bar(labels, avg_ms, color=colors, edgecolor="white", linewidth=1.2)

    # Add value labels on bars
    for bar, val, mt in zip(bars, avg_ms, mtypes):
        label = f"{val:.1f} ms"
        if mt != "real_hardware":
            label += "\nnon-real"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_ms) * 0.02,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Average latency (ms)", fontsize=12)
    ax.set_xlabel("Backend policy", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    if excluded:
        excluded_names = ", ".join(str(r.get("backend", "unknown")) for r in excluded)
        ax.text(
            0.99, 0.02,
            f"Excluded from bars: {excluded_names}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="gray",
        )

    out = output_path or (FIGURES_DIR / "backend_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


def plot_resource_usage(
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate resource usage gauge-style chart from resource_usage.csv."""
    rows = _read_csv("resource_usage.csv")
    if not rows:
        print("  [skip] resource_usage.csv is empty or missing")
        return None

    # Use the latest snapshot
    data = rows[-1]
    cpu = data.get("cpu_percent", 0)
    mem = data.get("memory_percent", 0)
    mem_mb = data.get("memory_used_mb", 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("System Resource Usage Snapshot (final sample)", fontsize=16, fontweight="bold")

    # CPU usage
    ax_cpu = axes[0]
    cpu_color = "#e15759" if cpu > 80 else "#f28e2b" if cpu > 50 else "#59a14f"
    ax_cpu.barh(["CPU"], [cpu], color=cpu_color, height=0.4)
    ax_cpu.barh(["CPU"], [100], color="#e0e0e0", height=0.4, zorder=0)
    ax_cpu.set_xlim(0, 100)
    ax_cpu.set_xlabel("Utilization (%)", fontsize=11)
    ax_cpu.set_title("CPU Utilization", fontsize=13)
    ax_cpu.text(cpu + 1, 0, f"{cpu:.1f}%", va="center", fontsize=12, fontweight="bold")

    # Memory usage
    ax_mem = axes[1]
    mem_color = "#e15759" if mem > 80 else "#f28e2b" if mem > 50 else "#59a14f"
    ax_mem.barh(["Memory"], [mem], color=mem_color, height=0.4)
    ax_mem.barh(["Memory"], [100], color="#e0e0e0", height=0.4, zorder=0)
    ax_mem.set_xlim(0, 100)
    ax_mem.set_xlabel("Utilization (%)", fontsize=11)
    ax_mem.set_title(f"Memory Utilization ({mem_mb:.0f} MB)", fontsize=13)
    ax_mem.text(mem + 1, 0, f"{mem:.1f}%", va="center", fontsize=12, fontweight="bold")

    out = output_path or (FIGURES_DIR / "resource_usage.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark charts from CSV results",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="CSV results directory (default: <project>/results/)",
    )
    parser.add_argument(
        "--figures-dir", type=str, default=None,
        help="Figure output directory (default: <project>/figures/)",
    )
    parser.add_argument(
        "--plots", type=str, nargs="+",
        default=["latency", "backend", "resource"],
        choices=["latency", "backend", "resource"],
        help="Plot types to generate (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    global RESULTS_DIR, FIGURES_DIR
    args = parse_args(argv)

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
    if args.figures_dir:
        FIGURES_DIR = Path(args.figures_dir)

    print("=" * 60)
    print("  Benchmark Result Plotting")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("=" * 60)

    _setup_plot_style()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if "latency" in args.plots:
        print("\n[1/3] Generating latency comparison ...")
        plot_latency_comparison()

    if "backend" in args.plots:
        print("\n[2/3] Generating backend comparison ...")
        plot_backend_comparison()

    if "resource" in args.plots:
        print("\n[3/3] Generating resource usage chart ...")
        plot_resource_usage()

    print("\n" + "=" * 60)
    print("  Plot generation complete.")
    print(f"  Figures saved in: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
