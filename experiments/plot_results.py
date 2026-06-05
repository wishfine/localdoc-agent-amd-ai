"""
绘图脚本 - Plot Benchmark Results

从 CSV 结果文件生成 matplotlib 图表：
- figures/latency_comparison.png   - 延迟对比图
- figures/backend_comparison.png   - 后端性能对比图
- figures/resource_usage.png       - 资源使用图

使用中文字体支持 (SimHei 优先，回退到默认字体)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Chinese font setup
# ---------------------------------------------------------------------------

def _setup_chinese_font() -> None:
    """Try to configure matplotlib for Chinese text rendering."""
    preferred_fonts = ["SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
                       "Microsoft YaHei", "PingFang SC", "STHeiti", "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"  [字体] 使用中文字体: {font_name}")
            return
    # Fallback: use default, Chinese chars may render as boxes
    print("  [字体] 警告: 未找到中文字体，中文字符可能无法正常显示。")
    print("         可执行 pip install matplotlib 并安装系统中文字体。")


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _read_csv(filename: str) -> List[Dict[str, Any]]:
    """Read a CSV file from RESULTS_DIR and return list of dicts."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"  [错误] 文件不存在: {filepath}")
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
    "CPU": "CPU (通用处理器)",
    "GPU": "GPU (图形处理器)",
    "NPU": "NPU (神经网络处理器)",
    "SimulatedNPU": "模拟 NPU",
}


def plot_latency_comparison(
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate latency comparison chart from latency_results.csv.

    Returns the path to the saved figure, or None if data is missing.
    """
    rows = _read_csv("latency_results.csv")
    if not rows:
        print("  [跳过] latency_results.csv 为空或不存在")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(
        "异构资源调度 - 延迟基准测试结果\n"
        "measurement_type 区分 real_hardware / simulated / unavailable",
        fontsize=14, fontweight="bold",
    )

    # Support both old (benchmark_latency) and new (benchmark_real) test names
    test_types = [
        ("embedding", "文档摄入/嵌入延迟", "文档数量", "doc_count"),
        ("query_embedding", "查询嵌入延迟", "Chunk 数量", "chunk_count"),
        ("end_to_end_rag", "端到端 RAG 延迟", "文档数量", "doc_count"),
    ]

    for ax, (test_key, title, xlabel, xkey) in zip(axes, test_types):
        subset = [r for r in rows if r.get("test") == test_key]
        # Filter out unavailable backends
        subset = [r for r in subset if r.get("measurement_type") != "unavailable"]
        if not subset:
            ax.set_title(f"{title} (无数据)")
            continue

        backends_in_data = sorted({r["backend"] for r in subset})
        for backend in backends_in_data:
            data = sorted(
                [r for r in subset if r["backend"] == backend],
                key=lambda r: r.get(xkey, 0),
            )
            x = [r.get(xkey, 0) for r in data]
            y = [r["latency_ms"] for r in data]
            ax.plot(x, y, "o-", label=BACKEND_LABELS.get(backend, backend),
                    color=BACKEND_COLORS.get(backend, None), linewidth=2, markersize=6)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("延迟 (ms)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    out = output_path or (FIGURES_DIR / "latency_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {out}")
    return out


def plot_backend_comparison(
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate backend performance bar chart from backend_results.csv."""
    rows = _read_csv("backend_results.csv")
    if not rows:
        print("  [跳过] backend_results.csv 为空或不存在")
        return None

    # Filter out unavailable backends
    rows = [r for r in rows if r.get("measurement_type") != "unavailable"]
    if not rows:
        print("  [跳过] backend_results.csv 无可用后端数据")
        return None

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.suptitle(
        "后端性能对比 - 平均延迟\n"
        "(unavailable 后端已排除)",
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

    ax.set_ylabel("平均延迟 (ms)", fontsize=12)
    ax.set_xlabel("后端策略", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    out = output_path or (FIGURES_DIR / "backend_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {out}")
    return out


def plot_resource_usage(
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate resource usage gauge-style chart from resource_usage.csv."""
    rows = _read_csv("resource_usage.csv")
    if not rows:
        print("  [跳过] resource_usage.csv 为空或不存在")
        return None

    # Use the latest snapshot
    data = rows[-1]
    cpu = data.get("cpu_percent", 0)
    mem = data.get("memory_percent", 0)
    mem_mb = data.get("memory_used_mb", 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("系统资源使用情况", fontsize=16, fontweight="bold")

    # CPU usage
    ax_cpu = axes[0]
    cpu_color = "#e15759" if cpu > 80 else "#f28e2b" if cpu > 50 else "#59a14f"
    ax_cpu.barh(["CPU"], [cpu], color=cpu_color, height=0.4)
    ax_cpu.barh(["CPU"], [100], color="#e0e0e0", height=0.4, zorder=0)
    ax_cpu.set_xlim(0, 100)
    ax_cpu.set_xlabel("使用率 (%)", fontsize=11)
    ax_cpu.set_title("CPU 使用率", fontsize=13)
    ax_cpu.text(cpu + 1, 0, f"{cpu:.1f}%", va="center", fontsize=12, fontweight="bold")

    # Memory usage
    ax_mem = axes[1]
    mem_color = "#e15759" if mem > 80 else "#f28e2b" if mem > 50 else "#59a14f"
    ax_mem.barh(["内存"], [mem], color=mem_color, height=0.4)
    ax_mem.barh(["内存"], [100], color="#e0e0e0", height=0.4, zorder=0)
    ax_mem.set_xlim(0, 100)
    ax_mem.set_xlabel("使用率 (%)", fontsize=11)
    ax_mem.set_title(f"内存使用率 ({mem_mb:.0f} MB)", fontsize=13)
    ax_mem.text(mem + 1, 0, f"{mem:.1f}%", va="center", fontsize=12, fontweight="bold")

    out = output_path or (FIGURES_DIR / "resource_usage.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 CSV 结果文件生成基准测试图表",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="CSV 结果目录 (默认: <project>/results/)",
    )
    parser.add_argument(
        "--figures-dir", type=str, default=None,
        help="图表输出目录 (默认: <project>/figures/)",
    )
    parser.add_argument(
        "--plots", type=str, nargs="+",
        default=["latency", "backend", "resource"],
        choices=["latency", "backend", "resource"],
        help="要生成的图表类型 (默认: 全部)",
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
    print("  基准测试结果绘图")
    print(f"  数据来源: {RESULTS_DIR}")
    print(f"  输出目录: {FIGURES_DIR}")
    print("=" * 60)

    _setup_chinese_font()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if "latency" in args.plots:
        print("\n[1/3] 生成延迟对比图 ...")
        plot_latency_comparison()

    if "backend" in args.plots:
        print("\n[2/3] 生成后端对比图 ...")
        plot_backend_comparison()

    if "resource" in args.plots:
        print("\n[3/3] 生成资源使用图 ...")
        plot_resource_usage()

    print("\n" + "=" * 60)
    print("  图表生成完成！")
    print(f"  图表保存在: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
