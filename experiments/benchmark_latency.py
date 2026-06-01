"""
延迟基准测试 - Simulated Latency Benchmark

⚠️ 这是一个 **模拟延迟基准测试**，不是真实 AMD 硬件实测。

使用 time.sleep() 和后端延迟乘数模拟不同后端策略的延迟差异。
所有数据标记为 is_simulated=True，measurement_type="simulated_latency"。

目的：验证异构调度框架在不同后端策略下的行为差异，
不代表真实 AMD GPU/NPU 的性能。

输出 CSV:
- results/latency_results.csv
- results/backend_results.csv
- results/resource_usage.csv
"""

import argparse
import csv
import os
import random
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# psutil is optional - gracefully degrade if not installed
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Default test parameters
DEFAULT_DOC_COUNTS = [1, 5, 10, 20, 50]
DEFAULT_CHUNK_COUNTS = [10, 50, 100, 500]
DEFAULT_BACKENDS = ["CPU", "GPU", "NPU", "SimulatedNPU"]
DEFAULT_REPEATS = 3

# Synthetic document templates (Chinese + English mix)
SECTION_TEMPLATES = [
    "# {title}\n\n## 概述\n\n{content}\n\n## 详细说明\n\n{detail}\n",
    "## {title}\n\n{content}\n\n### 背景\n\n{detail}\n\n### 结论\n\n{conclusion}\n",
    "# {title}\n\n> 摘要: {content}\n\n## 正文\n\n{detail}\n",
]

TOPICS = [
    "异构计算架构分析", "AMD 锐龙 AI 处理器技术白皮书",
    "本地知识库智能体设计方案", "RAG 检索增强生成系统",
    "NPU 推理加速方案", "文档分块策略对比研究",
    "向量数据库选型指南", "嵌入模型性能评估",
    "混合精度推理优化", "端侧大语言模型部署",
    "Heterogeneous Computing Overview", "Local Knowledge Base Agent Design",
    "RAG Pipeline Architecture", "NPU Acceleration Strategies",
    "Document Chunking Methods", "Embedding Model Benchmarking",
]


def _random_text(min_sentences: int = 3, max_sentences: int = 8) -> str:
    """Generate a block of random Chinese-style placeholder text."""
    sentences = []
    for _ in range(random.randint(min_sentences, max_sentences)):
        length = random.randint(10, 40)
        sentence = "".join(random.choices(string.ascii_lowercase + "  ,.", k=length))
        sentences.append(sentence.capitalize() + ".")
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_test_documents(count: int, size: int = 500) -> List[str]:
    """Generate *count* synthetic markdown documents, each roughly *size* chars.

    Returns a list of markdown-formatted strings.
    """
    docs: List[str] = []
    for i in range(count):
        topic = random.choice(TOPICS)
        template = random.choice(SECTION_TEMPLATES)
        doc = template.format(
            title=f"{topic} - 第{i + 1}节",
            content=_random_text(4, 8),
            detail=_random_text(6, 12),
            conclusion=_random_text(2, 4),
        )
        # Pad to approximately the requested size
        while len(doc) < size:
            doc += "\n\n" + _random_text(4, 8)
        docs.append(doc[:size])
    return docs


# ---------------------------------------------------------------------------
# Timing / resource helpers
# ---------------------------------------------------------------------------

def _time_call(func, *args, repeats: int = DEFAULT_REPEATS, **kwargs) -> Tuple[float, Any]:
    """Run *func* multiple times and return (average_seconds, last_result)."""
    total = 0.0
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        total += time.perf_counter() - start
    return total / repeats, result


def collect_resource_usage() -> Dict[str, float]:
    """Collect current CPU / memory usage.  Returns a dict with percentages."""
    if not HAS_PSUTIL:
        return {"cpu_percent": -1.0, "memory_percent": -1.0, "memory_used_mb": -1.0}
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": cpu,
        "memory_percent": mem.percent,
        "memory_used_mb": mem.used / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Simulated backend operations
# ---------------------------------------------------------------------------

# Each backend has a characteristic latency multiplier to make benchmarks
# realistic without requiring actual hardware.
_BACKEND_MULTIPLIER: Dict[str, float] = {
    "CPU": 1.0,
    "GPU": 0.6,
    "NPU": 0.3,
    "SimulatedNPU": 0.45,
}


def _simulate_ingest(doc: str, backend: str) -> int:
    """Simulate ingesting a single document.  Returns chunk count."""
    multiplier = _BACKEND_MULTIPLIER.get(backend, 1.0)
    # Simulate work proportional to document length
    work = len(doc) * multiplier * 1e-5
    time.sleep(work)
    # Return a pseudo chunk count
    return max(1, len(doc) // 200)


def _simulate_query(query: str, chunk_count: int, backend: str) -> str:
    """Simulate a query against *chunk_count* stored chunks."""
    multiplier = _BACKEND_MULTIPLIER.get(backend, 1.0)
    work = chunk_count * multiplier * 2e-5
    time.sleep(work)
    return f"[{backend}] 模拟回答: 找到 {min(3, chunk_count)} 个相关片段。"


def _simulate_end_to_end(docs: List[str], query: str, backend: str) -> str:
    """Simulate the full ingest-then-query pipeline."""
    for doc in docs:
        _simulate_ingest(doc, backend)
    chunk_count = sum(max(1, len(d) // 200) for d in docs)
    return _simulate_query(query, chunk_count, backend)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_ingestion(
    doc_counts: List[int],
    backends: List[str],
    repeats: int = DEFAULT_REPEATS,
) -> List[Dict[str, Any]]:
    """Benchmark document ingestion latency for each (doc_count, backend) pair."""
    rows: List[Dict[str, Any]] = []
    for count in doc_counts:
        docs = generate_test_documents(count)
        for backend in backends:
            elapsed, _ = _time_call(
                lambda d=docs, b=backend: [_simulate_ingest(doc, b) for doc in d],
                repeats=repeats,
            )
            resources = collect_resource_usage()
            row = {
                "test": "ingestion",
                "doc_count": count,
                "backend": backend,
                "latency_s": round(elapsed, 6),
                "latency_ms": round(elapsed * 1000, 3),
                "is_simulated": True,
                "measurement_type": "simulated_latency",
                "note": "Simulated backend latency, not real AMD hardware measurement.",
                **resources,
            }
            rows.append(row)
            print(f"  [ingestion] docs={count:>3}, simulated {backend:<14} policy -> {elapsed * 1000:>8.1f} ms")
    return rows


def benchmark_querying(
    chunk_counts: List[int],
    backends: List[str],
    repeats: int = DEFAULT_REPEATS,
) -> List[Dict[str, Any]]:
    """Benchmark query latency for each (chunk_count, backend) pair."""
    query = "请总结异构计算架构的核心优势。"
    rows: List[Dict[str, Any]] = []
    for count in chunk_counts:
        for backend in backends:
            elapsed, _ = _time_call(
                lambda q=query, c=count, b=backend: _simulate_query(q, c, b),
                repeats=repeats,
            )
            resources = collect_resource_usage()
            row = {
                "test": "querying",
                "chunk_count": count,
                "backend": backend,
                "latency_s": round(elapsed, 6),
                "latency_ms": round(elapsed * 1000, 3),
                "is_simulated": True,
                "measurement_type": "simulated_latency",
                "note": "Simulated backend latency, not real AMD hardware measurement.",
                **resources,
            }
            rows.append(row)
            print(f"  [query] chunks={count:>4}, simulated {backend:<14} policy -> {elapsed * 1000:>8.1f} ms")
    return rows


def benchmark_end_to_end(
    configs: List[Tuple[int, str]],
    repeats: int = DEFAULT_REPEATS,
) -> List[Dict[str, Any]]:
    """Benchmark full pipeline (ingest + query) for each config.

    *configs* is a list of (doc_count, backend) tuples.
    """
    query = "请总结异构计算架构的核心优势。"
    rows: List[Dict[str, Any]] = []
    for doc_count, backend in configs:
        docs = generate_test_documents(doc_count)
        elapsed, _ = _time_call(
            lambda d=docs, q=query, b=backend: _simulate_end_to_end(d, q, b),
            repeats=repeats,
        )
        resources = collect_resource_usage()
        row = {
            "test": "end_to_end",
            "doc_count": doc_count,
            "backend": backend,
            "latency_s": round(elapsed, 6),
            "latency_ms": round(elapsed * 1000, 3),
            "is_simulated": True,
            "measurement_type": "simulated_latency",
            "note": "Simulated backend latency, not real AMD hardware measurement.",
            **resources,
        }
        rows.append(row)
        print(f"  [end-to-end] docs={doc_count:>3}, simulated {backend:<14} policy -> {elapsed * 1000:>8.1f} ms")
    return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_results_csv(results: List[Dict[str, Any]], filename: str) -> Path:
    """Write *results* (list of dicts) to ``RESULTS_DIR / filename``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename
    if not results:
        print(f"  [警告] 无结果可写入 {filepath.name}")
        return filepath
    # Collect the union of all keys so rows with different schemas can coexist
    all_keys: list = []
    seen: set = set()
    for row in results:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    fieldnames = all_keys
    # Fill missing keys with empty string so every row has the same schema
    normalised = [{k: row.get(k, "") for k in fieldnames} for row in results]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalised)
    print(f"  [保存] {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="延迟基准测试 - 测试不同配置下的系统响应时间",
    )
    parser.add_argument(
        "--doc-counts", type=int, nargs="+", default=DEFAULT_DOC_COUNTS,
        help="要测试的文档数量列表 (默认: 1 5 10 20 50)",
    )
    parser.add_argument(
        "--chunk-counts", type=int, nargs="+", default=DEFAULT_CHUNK_COUNTS,
        help="要测试的 chunk 数量列表 (默认: 10 50 100 500)",
    )
    parser.add_argument(
        "--backends", type=str, nargs="+", default=DEFAULT_BACKENDS,
        choices=DEFAULT_BACKENDS,
        help="要测试的后端策略 (默认: CPU GPU NPU SimulatedNPU)",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help="每个测试重复次数，取平均值 (默认: 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认: <project>/results/)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    global RESULTS_DIR
    if args.output_dir:
        RESULTS_DIR = Path(args.output_dir)

    print("=" * 60)
    print("  Simulated Latency Benchmark (模拟延迟基准测试)")
    print("  ⚠️ NOT real AMD hardware measurement")
    print(f"  文档数量: {args.doc_counts}")
    print(f"  Chunk数量: {args.chunk_counts}")
    print(f"  后端策略: {args.backends} (all simulated)")
    print(f"  重复次数: {args.repeats}")
    print(f"  psutil 可用: {HAS_PSUTIL}")
    print("=" * 60)

    # --- Ingestion benchmark ---
    print("\n[1/3] 运行文档摄入基准测试 ...")
    ingestion_results = benchmark_ingestion(args.doc_counts, args.backends, args.repeats)

    # --- Query benchmark ---
    print("\n[2/3] 运行查询基准测试 ...")
    query_results = benchmark_querying(args.chunk_counts, args.backends, args.repeats)

    # --- End-to-end benchmark ---
    print("\n[3/3] 运行端到端基准测试 ...")
    e2e_configs = [(dc, b) for dc in args.doc_counts for b in args.backends]
    e2e_results = benchmark_end_to_end(e2e_configs, args.repeats)

    # --- Save ---
    print("\n[保存] 写入 CSV 结果 ...")
    latency_rows = ingestion_results + query_results + e2e_results
    save_results_csv(latency_rows, "latency_results.csv")

    # Build a backend-only summary
    backend_summary: List[Dict[str, Any]] = []
    for backend in args.backends:
        subset = [r for r in latency_rows if r["backend"] == backend]
        if subset:
            avg_ms = sum(r["latency_ms"] for r in subset) / len(subset)
            backend_summary.append({
                "backend": backend,
                "avg_latency_ms": round(avg_ms, 3),
                "test_count": len(subset),
                "is_simulated": True,
                "measurement_type": "simulated_latency",
                "note": "Simulated backend policy comparison, not real hardware.",
            })
    save_results_csv(backend_summary, "backend_results.csv")

    # Resource usage snapshot
    resource_snapshot = collect_resource_usage()
    save_results_csv([resource_snapshot], "resource_usage.csv")

    print("\n" + "=" * 60)
    print("  基准测试完成！")
    print(f"  结果保存在: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
