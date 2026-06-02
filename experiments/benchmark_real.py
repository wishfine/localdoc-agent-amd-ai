"""
Real Hardware Benchmark (with simulated fallback)

When real AMD GPU/NPU hardware is detected, benchmarks use actual backend
methods (embed_texts, generate_answer). When no hardware is available,
falls back to simulated mode with clear labeling.

This replaces the pure-simulated benchmark_latency.py for hardware-capable
environments. On CPU-only machines, it produces the same simulated data
but with explicit "measurement_type" column distinguishing real from sim.

Output:
- results/latency_results.csv (unified: real or simulated per row)
- results/backend_results.csv (summary with measurement_type)
- results/resource_usage.csv

Usage:
    python experiments/benchmark_real.py
    python experiments/benchmark_real.py --mode simulated   # force simulated
    python experiments/benchmark_real.py --mode real        # force real (fails if no HW)
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Synthetic document generation (shared with benchmark_latency.py)
# ---------------------------------------------------------------------------

_PLACEHOLDER_SENTENCES = [
    "异构计算架构通过组合不同类型的处理单元来优化系统性能。",
    "CPU 适合复杂控制逻辑和串行任务处理。",
    "GPU 擅长大规模并行浮点运算和矩阵计算。",
    "NPU 专为低功耗神经网络推理而设计。",
    "AMD 锐龙 AI MAX+ 处理器集成了三种计算单元。",
    "合理的任务调度策略能够最大化异构硬件的利用率。",
    "本地知识库智能体将文档处理和推理完全放在本地完成。",
    "向量嵌入技术将文本转换为数值表示用于语义检索。",
    "余弦相似度是衡量向量方向接近程度的经典方法。",
    "TF-IDF 通过词频和逆文档频率衡量词语的重要性。",
    "文本切块策略需要平衡语义完整性和检索粒度。",
    "异构调度器根据任务计算特征选择最优硬件后端。",
    "ROCm 是 AMD 的 GPU 计算平台，支持 HIP 编程模型。",
    "XDNA 架构的 NPU 提供高效的端侧 AI 推理能力。",
    "统一内存架构使 CPU、GPU、NPU 可以共享系统内存。",
]


def generate_test_documents(count: int, size: int = 500) -> List[str]:
    docs = []
    for i in range(count):
        doc = "".join(random.choices(_PLACEHOLDER_SENTENCES, k=max(1, size // 30)))
        docs.append(doc[:size])
    return docs


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_backends() -> Dict[str, Any]:
    """Detect available real backends. Returns dict of backend instances."""
    from localdoc.backends.cpu_backend import CPUBackend

    backends = {"CPU": CPUBackend()}

    # Try GPU
    try:
        from localdoc.backends.gpu_backend import AMDGPUBackend
        gpu = AMDGPUBackend()
        if gpu.is_available():
            backends["GPU"] = gpu
    except Exception:
        pass

    # Try NPU
    try:
        from localdoc.backends.npu_backend import AMDNPUBackend
        npu = AMDNPUBackend()
        if npu.is_available():
            backends["NPU"] = npu
    except Exception:
        pass

    # Always add simulated NPU for comparison
    try:
        from localdoc.backends.simulated_npu import SimulatedNPUBackend
        backends["SimulatedNPU"] = SimulatedNPUBackend()
    except Exception:
        pass

    return backends


# ---------------------------------------------------------------------------
# Real benchmark (uses actual backend methods)
# ---------------------------------------------------------------------------

def benchmark_real_ingest(
    doc_counts: List[int],
    backends: Dict[str, Any],
    repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Benchmark real document ingestion using actual backend embed_texts."""
    from localdoc.backends.cpu_backend import CPUBackend

    rows = []
    for count in doc_counts:
        docs = generate_test_documents(count)
        all_texts = []
        for doc in docs:
            # Simple chunking: split by sentences
            import re
            sentences = re.split(r'[。！？.!?\n]+', doc)
            all_texts.extend([s.strip() for s in sentences if s.strip()])

        for backend_name, backend in backends.items():
            if backend_name == "SimulatedNPU":
                continue  # Skip simulated in real benchmark

            total_time = 0.0
            for _ in range(repeats):
                # Reset corpus for CPUBackend
                if hasattr(backend, 'reset_corpus'):
                    backend.reset_corpus()

                t0 = time.perf_counter()
                if hasattr(backend, 'fit_and_embed'):
                    backend.fit_and_embed(all_texts)
                elif hasattr(backend, 'embed_texts'):
                    backend.embed_texts(all_texts)
                total_time += time.perf_counter() - t0

            avg_time = total_time / repeats

            # Get hardware info
            is_sim = backend_name == "SimulatedNPU"
            device_info = {}
            if hasattr(backend, 'get_device_info'):
                device_info = backend.get_device_info()

            rows.append({
                "test": "ingestion",
                "doc_count": count,
                "backend": backend_name,
                "latency_s": round(avg_time, 6),
                "latency_ms": round(avg_time * 1000, 3),
                "measurement_type": "simulated" if is_sim else "real_hardware",
                "is_simulated": is_sim,
                "device": device_info.get("device", "cpu"),
                "note": "Simulated backend" if is_sim else "Real hardware measurement",
            })
            print(f"  [ingest] docs={count:>3}, {backend_name:<14} -> {avg_time * 1000:>8.1f} ms "
                  f"({'simulated' if is_sim else 'REAL'})")

    return rows


def benchmark_real_query(
    chunk_counts: List[int],
    backends: Dict[str, Any],
    repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Benchmark real query using actual backend transform/generate."""
    rows = []
    query = "请总结异构计算架构的核心优势。"

    for count in chunk_counts:
        # Prepare corpus
        corpus = generate_test_documents(count, size=200)
        import re
        all_texts = []
        for doc in corpus:
            sentences = re.split(r'[。！？.!?\n]+', doc)
            all_texts.extend([s.strip() for s in sentences if s.strip()][:3])

        for backend_name, backend in backends.items():
            if backend_name == "SimulatedNPU":
                continue

            # Fit on corpus first
            if hasattr(backend, 'reset_corpus'):
                backend.reset_corpus()
            if hasattr(backend, 'fit_and_embed'):
                backend.fit_and_embed(all_texts[:count])

            total_time = 0.0
            for _ in range(repeats):
                t0 = time.perf_counter()
                if hasattr(backend, 'transform'):
                    backend.transform([query])
                elif hasattr(backend, 'embed_texts'):
                    backend.embed_texts([query])
                total_time += time.perf_counter() - t0

            avg_time = total_time / repeats
            is_sim = backend_name == "SimulatedNPU"

            rows.append({
                "test": "querying",
                "chunk_count": count,
                "backend": backend_name,
                "latency_s": round(avg_time, 6),
                "latency_ms": round(avg_time * 1000, 3),
                "measurement_type": "simulated" if is_sim else "real_hardware",
                "is_simulated": is_sim,
                "note": "Simulated backend" if is_sim else "Real hardware measurement",
            })
            print(f"  [query] chunks={count:>4}, {backend_name:<14} -> {avg_time * 1000:>8.1f} ms "
                  f"({'simulated' if is_sim else 'REAL'})")

    return rows


# ---------------------------------------------------------------------------
# Simulated benchmark (for comparison / CPU-only machines)
# ---------------------------------------------------------------------------

_BACKEND_MULTIPLIER = {"CPU": 1.0, "GPU": 0.6, "NPU": 0.3, "SimulatedNPU": 0.45}


def benchmark_simulated(
    doc_counts: List[int],
    chunk_counts: List[int],
    repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Pure simulated benchmark using time.sleep()."""
    rows = []

    for count in doc_counts:
        docs = generate_test_documents(count)
        for backend, mult in _BACKEND_MULTIPLIER.items():
            total = 0.0
            for _ in range(repeats):
                t0 = time.perf_counter()
                for doc in docs:
                    time.sleep(len(doc) * mult * 1e-5)
                total += time.perf_counter() - t0
            avg = total / repeats
            rows.append({
                "test": "ingestion",
                "doc_count": count,
                "backend": backend,
                "latency_s": round(avg, 6),
                "latency_ms": round(avg * 1000, 3),
                "measurement_type": "simulated",
                "is_simulated": True,
                "note": "Simulated via time.sleep multiplier, not real hardware",
            })
            print(f"  [ingest] docs={count:>3}, simulated {backend:<14} -> {avg * 1000:>8.1f} ms")

    for count in chunk_counts:
        for backend, mult in _BACKEND_MULTIPLIER.items():
            total = 0.0
            for _ in range(repeats):
                t0 = time.perf_counter()
                time.sleep(count * mult * 2e-5)
                total += time.perf_counter() - t0
            avg = total / repeats
            rows.append({
                "test": "querying",
                "chunk_count": count,
                "backend": backend,
                "latency_s": round(avg, 6),
                "latency_ms": round(avg * 1000, 3),
                "measurement_type": "simulated",
                "is_simulated": True,
                "note": "Simulated via time.sleep multiplier, not real hardware",
            })
            print(f"  [query] chunks={count:>4}, simulated {backend:<14} -> {avg * 1000:>8.1f} ms")

    return rows


# ---------------------------------------------------------------------------
# Resource usage
# ---------------------------------------------------------------------------

def collect_resource_usage() -> Dict[str, float]:
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
# CSV helpers
# ---------------------------------------------------------------------------

def save_csv(rows: List[Dict[str, Any]], filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename
    if not rows:
        return filepath
    all_keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in all_keys} for row in rows])
    print(f"  [saved] {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real hardware benchmark with simulated fallback")
    parser.add_argument("--mode", choices=["auto", "real", "simulated"], default="auto")
    parser.add_argument("--doc-counts", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--chunk-counts", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("  Real Hardware Benchmark")
    print("=" * 60)

    # Detect hardware
    backends = detect_backends()
    has_real_gpu = "GPU" in backends
    has_real_npu = "NPU" in backends
    has_real_hw = has_real_gpu or has_real_npu

    print(f"\n  Detected backends: {list(backends.keys())}")
    print(f"  Real GPU: {has_real_gpu}")
    print(f"  Real NPU: {has_real_npu}")

    # Decide mode
    if args.mode == "auto":
        use_real = has_real_hw
    elif args.mode == "real":
        if not has_real_hw:
            print("\n  ❌ --mode=real requested but no real GPU/NPU detected.")
            print("  Run: python experiments/check_environment.py")
            sys.exit(1)
        use_real = True
    else:
        use_real = False

    mode_str = "REAL HARDWARE" if use_real else "SIMULATED"
    print(f"\n  Mode: {mode_str}")
    print(f"  Doc counts: {args.doc_counts}")
    print(f"  Chunk counts: {args.chunk_counts}")
    print(f"  Repeats: {args.repeats}")
    print("=" * 60)

    # Run benchmarks
    if use_real:
        print("\n[1/2] Running REAL hardware benchmarks ...")
        ingest_rows = benchmark_real_ingest(args.doc_counts, backends, args.repeats)
        query_rows = benchmark_real_query(args.chunk_counts, backends, args.repeats)
        # Also run simulated for comparison
        print("\n[2/2] Running simulated benchmarks for comparison ...")
        sim_rows = benchmark_simulated(args.doc_counts, args.chunk_counts, args.repeats)
        all_rows = ingest_rows + query_rows + sim_rows
    else:
        print("\n[1/1] Running simulated benchmarks ...")
        all_rows = benchmark_simulated(args.doc_counts, args.chunk_counts, args.repeats)

    # Save results
    print("\n[save] Writing CSV ...")
    save_csv(all_rows, "latency_results.csv")

    # Backend summary
    summary = []
    for backend in set(r["backend"] for r in all_rows):
        subset = [r for r in all_rows if r["backend"] == backend]
        real_rows = [r for r in subset if not r.get("is_simulated", True)]
        sim_rows = [r for r in subset if r.get("is_simulated", True)]

        if real_rows:
            avg_real = sum(r["latency_ms"] for r in real_rows) / len(real_rows)
            summary.append({
                "backend": backend,
                "avg_latency_ms": round(avg_real, 3),
                "test_count": len(real_rows),
                "measurement_type": "real_hardware",
                "is_simulated": False,
                "note": "Real hardware measurement",
            })
        if sim_rows:
            avg_sim = sum(r["latency_ms"] for r in sim_rows) / len(sim_rows)
            summary.append({
                "backend": backend,
                "avg_latency_ms": round(avg_sim, 3),
                "test_count": len(sim_rows),
                "measurement_type": "simulated",
                "is_simulated": True,
                "note": "Simulated via time.sleep multiplier",
            })

    save_csv(summary, "backend_results.csv")
    save_csv([collect_resource_usage()], "resource_usage.csv")

    print("\n" + "=" * 60)
    print(f"  Benchmark complete! Mode: {mode_str}")
    if not use_real:
        print("  ⚠️ All results are SIMULATED. Not real AMD hardware data.")
        print("  To get real data, run on AMD Ryzen AI MAX+ hardware.")
    print("=" * 60)


if __name__ == "__main__":
    main()
