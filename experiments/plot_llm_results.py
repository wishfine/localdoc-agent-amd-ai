"""
Plot LLM Benchmark Results

Generates charts from LLM benchmark CSVs.
All charts are titled as "Local LLM benchmark, not AMD GPU/NPU hardware benchmark".

Output:
- figures/llm_generation_latency.png
- figures/rag_mode_comparison.png
- figures/rag_stage_breakdown.png
"""

import csv
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def _setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _read_csv(filename: str) -> List[Dict[str, Any]]:
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return []
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
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


def plot_llm_generation_latency(output_path: Optional[Path] = None) -> Optional[Path]:
    rows = _read_csv("llm_generation_benchmark.csv")
    if not rows or all(r.get("query_id") == "SKIPPED" for r in rows):
        print("  [skip] llm_generation_benchmark.csv has no data")
        return None

    # Filter out SKIPPED rows
    data = [r for r in rows if r.get("query_id") != "SKIPPED"]
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Local LLM Generation Latency (Qwen3-1.7B)\n"
        "Local LLM benchmark, not AMD GPU/NPU hardware benchmark",
        fontsize=13, fontweight="bold",
    )

    queries = [f"Q{r['query_id']}" for r in data]
    gen_times = [r["generation_time_s"] for r in data]
    tps = [r.get("tokens_per_second", 0) for r in data]

    x = range(len(queries))
    bars = ax.bar(x, gen_times, color="#4e79a7", edgecolor="white")
    for bar, val in zip(bars, gen_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=10)

    ax2 = ax.twinx()
    ax2.plot(x, tps, "ro-", markersize=8, label="tokens/s")
    ax2.set_ylabel("Tokens/s", fontsize=11, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax.set_xticks(x)
    ax.set_xticklabels(queries, fontsize=10)
    ax.set_ylabel("Generation Time (s)", fontsize=11)
    ax.set_xlabel("Query", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    device = data[0].get("device", "unknown")
    ax.annotate(f"Device: {device}", xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=9, color="gray")

    out = output_path or (FIGURES_DIR / "llm_generation_latency.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


def plot_rag_mode_comparison(output_path: Optional[Path] = None) -> Optional[Path]:
    rows = _read_csv("rag_mode_comparison.csv")
    if not rows:
        print("  [skip] rag_mode_comparison.csv has no data")
        return None

    data = [r for r in rows if r.get("total_time_s") != "SKIPPED"]
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "RAG Mode Comparison (Extractive vs Local LLM)\n"
        "Local LLM benchmark, not AMD GPU/NPU hardware benchmark",
        fontsize=13, fontweight="bold",
    )

    modes = [r["mode"] for r in data]
    ingest = [r["ingest_time_s"] for r in data]
    query = [r["query_time_s"] for r in data]

    x = range(len(modes))
    width = 0.35
    ax.bar([i - width / 2 for i in x], ingest, width, label="Ingest", color="#4e79a7")
    ax.bar([i + width / 2 for i in x], query, width, label="Query/Generate", color="#f28e2b")

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=10)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = output_path or (FIGURES_DIR / "rag_mode_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


def plot_rag_stage_breakdown(output_path: Optional[Path] = None) -> Optional[Path]:
    rows = _read_csv("rag_stage_breakdown.csv")
    if not rows:
        print("  [skip] rag_stage_breakdown.csv has no data")
        return None

    data = [r for r in rows if r.get("time_s") != "SKIPPED"]
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "RAG Coarse Latency Breakdown (ingest / query)\n"
        "Local LLM benchmark, not AMD GPU/NPU hardware benchmark",
        fontsize=13, fontweight="bold",
    )

    modes = list(dict.fromkeys(r["mode"] for r in data))
    stages = ["ingest", "query"]

    for i, mode in enumerate(modes):
        vals = []
        for stage in stages:
            match = [r for r in data if r["mode"] == mode and r["stage"] == stage]
            vals.append(match[0]["time_s"] if match else 0)

        x_pos = [i - 0.15, i + 0.15]
        bars = ax.bar(x_pos, vals, 0.3, label=mode if i == 0 else "")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.2f}s", ha="center", fontsize=9)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, fontsize=10)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    out = output_path or (FIGURES_DIR / "rag_stage_breakdown.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local LLM benchmark charts")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR))
    return parser.parse_args()


def main():
    global RESULTS_DIR, FIGURES_DIR
    args = parse_args()
    RESULTS_DIR = Path(args.results_dir)
    FIGURES_DIR = Path(args.figures_dir)

    print("=" * 60)
    print("  LLM Benchmark Plotting")
    print("  Local LLM benchmark, not AMD hardware benchmark")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("=" * 60)

    _setup_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Generating LLM generation latency chart ...")
    plot_llm_generation_latency()

    print("\n[2/3] Generating RAG mode comparison chart ...")
    plot_rag_mode_comparison()

    print("\n[3/3] Generating RAG latency breakdown chart ...")
    plot_rag_stage_breakdown()

    print("\nPlot generation complete.")


if __name__ == "__main__":
    main()
