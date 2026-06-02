"""
Real Hardware Benchmark (with simulated fallback)

When real AMD GPU/NPU hardware is detected AND the backend performs real
inference (not just CPU fallback), benchmarks use actual backend methods.
Otherwise falls back to simulated mode with clear labeling.

Key distinction:
- is_available() = EP/driver detected (necessary but not sufficient)
- has_real_inference() = backend actually runs computation on that hardware

For NPU: current implementation does NumPy normalization even when EP is
detected, so has_real_inference() returns False until real ONNX session
is implemented.

Output:
- results/latency_results.csv
- results/backend_results.csv
- results/resource_usage.csv
"""

import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Synthetic data
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
]


def generate_test_documents(count: int, size: int = 500) -> List[str]:
    docs = []
    for i in range(count):
        doc = "".join(random.choices(_PLACEHOLDER_SENTENCES, k=max(1, size // 30)))
        docs.append(doc[:size])
    return docs


def chunk_texts(texts: List[str]) -> List[str]:
    """Simple sentence-level chunking."""
    chunks = []
    for text in texts:
        sentences = re.split(r'[。！？.!?\n]+', text)
        chunks.extend([s.strip() for s in sentences if s.strip()])
    return chunks


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_backends() -> Dict[str, Dict[str, Any]]:
    """
    Detect available backends and their real inference status.

    Returns dict: name -> {"backend": instance, "available": bool, "real_inference": bool}
    """
    from localdoc.backends.cpu_backend import CPUBackend

    result = {"CPU": {"backend": CPUBackend(), "available": True, "real_inference": True}}

    try:
        from localdoc.backends.gpu_backend import AMDGPUBackend
        gpu = AMDGPUBackend()
        avail = gpu.is_available()
        real = gpu.has_real_inference() if hasattr(gpu, 'has_real_inference') else avail
        result["GPU"] = {"backend": gpu, "available": avail, "real_inference": real}
    except Exception:
        pass

    try:
        from localdoc.backends.npu_backend import AMDNPUBackend
        npu = AMDNPUBackend()
        avail = npu.is_available()
        real = npu.has_real_inference() if hasattr(npu, 'has_real_inference') else False
        result["NPU"] = {"backend": npu, "available": avail, "real_inference": real}
    except Exception:
        pass

    try:
        from localdoc.backends.simulated_npu import SimulatedNPUBackend
        result["SimulatedNPU"] = {
            "backend": SimulatedNPUBackend(),
            "available": True,
            "real_inference": False,
        }
    except Exception:
        pass

    return result


def classify_backend(name: str, info: Dict[str, Any]) -> str:
    """Return measurement_type string for a backend."""
    if name == "SimulatedNPU":
        return "simulated"
    if not info["available"]:
        return "unavailable"
    if info["real_inference"]:
        return "real_hardware"
    # Available but no real inference (e.g., NPU EP detected but NumPy fallback)
    return "cpu_fallback_with_hardware_detected"


def _skip_backend(name: str, info: Dict[str, Any]) -> bool:
    """Return True if backend should be skipped (unavailable)."""
    if name == "SimulatedNPU":
        return False  # Always include for comparison
    return not info["available"]


# ---------------------------------------------------------------------------
# Real benchmark: embedding
# ---------------------------------------------------------------------------

def benchmark_embedding(
    doc_counts: List[int],
    backends: Dict[str, Dict[str, Any]],
    repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Benchmark embedding using actual backend methods."""
    rows = []
    for count in doc_counts:
        texts = chunk_texts(generate_test_documents(count))

        for name, info in backends.items():
            if _skip_backend(name, info):
                continue
            backend = info["backend"]
            mtype = classify_backend(name, info)

            total = 0.0
            for _ in range(repeats):
                if hasattr(backend, 'reset_corpus'):
                    backend.reset_corpus()
                t0 = time.perf_counter()
                if hasattr(backend, 'fit_and_embed'):
                    backend.fit_and_embed(texts)
                elif hasattr(backend, 'embed_texts'):
                    backend.embed_texts(texts)
                total += time.perf_counter() - t0

            avg = total / repeats
            rows.append({
                "test": "embedding",
                "doc_count": count,
                "chunk_count": len(texts),
                "backend": name,
                "latency_s": round(avg, 6),
                "latency_ms": round(avg * 1000, 3),
                "measurement_type": mtype,
                "is_simulated": mtype == "simulated",
                "real_inference": info["real_inference"],
                "note": _note_for(mtype),
            })
            print(f"  [embed] docs={count:>3}, {name:<14} -> {avg * 1000:>8.1f} ms  [{mtype}]")

    return rows


# ---------------------------------------------------------------------------
# Real benchmark: query (transform)
# ---------------------------------------------------------------------------

def benchmark_query(
    chunk_counts: List[int],
    backends: Dict[str, Dict[str, Any]],
    repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Benchmark query embedding using actual backend transform."""
    query = "请总结异构计算架构的核心优势。"
    rows = []

    for count in chunk_counts:
        corpus = chunk_texts(generate_test_documents(5, size=200))[:count]

        for name, info in backends.items():
            if _skip_backend(name, info):
                continue
            backend = info["backend"]
            mtype = classify_backend(name, info)

            # Fit on corpus
            if hasattr(backend, 'reset_corpus'):
                backend.reset_corpus()
            if hasattr(backend, 'fit_and_embed'):
                backend.fit_and_embed(corpus)
            elif hasattr(backend, 'embed_texts'):
                backend.embed_texts(corpus)

            # Benchmark query
            total = 0.0
            for _ in range(repeats):
                t0 = time.perf_counter()
                if hasattr(backend, 'transform'):
                    backend.transform([query])
                elif hasattr(backend, 'embed_texts'):
                    backend.embed_texts([query])
                total += time.perf_counter() - t0

            avg = total / repeats
            rows.append({
                "test": "query_embedding",
                "chunk_count": count,
                "backend": name,
                "latency_s": round(avg, 6),
                "latency_ms": round(avg * 1000, 3),
                "measurement_type": mtype,
                "is_simulated": mtype == "simulated",
                "real_inference": info["real_inference"],
                "note": _note_for(mtype),
            })
            print(f"  [query_embed] chunks={count:>4}, {name:<14} -> {avg * 1000:>8.1f} ms  [{mtype}]")

    return rows


# ---------------------------------------------------------------------------
# Real benchmark: generation (generate_answer)
# ---------------------------------------------------------------------------

def benchmark_generation(
    backends: Dict[str, Dict[str, Any]],
    repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Benchmark answer generation using actual backend generate_answer."""
    query = "什么是异构计算？"
    context_chunks = [
        "异构计算是指在同一个计算系统中同时使用不同类型的处理单元。",
        "CPU 适合复杂控制逻辑，GPU 适合并行计算，NPU 适合推理。",
        "AMD 锐龙 AI MAX+ 集成了 CPU、GPU 和 NPU 三种计算单元。",
    ]
    context_str = "\n".join(context_chunks)

    rows = []
    for name, info in backends.items():
        if _skip_backend(name, info):
            continue
        backend = info["backend"]
        mtype = classify_backend(name, info)

        if not hasattr(backend, 'generate_answer'):
            continue

        total = 0.0
        answer = ""
        for _ in range(repeats):
            t0 = time.perf_counter()
            answer = backend.generate_answer(query=query, context=context_str)
            total += time.perf_counter() - t0

        avg = total / repeats
        rows.append({
            "test": "generation",
            "backend": name,
            "latency_s": round(avg, 6),
            "latency_ms": round(avg * 1000, 3),
            "answer_length": len(answer),
            "measurement_type": mtype,
            "is_simulated": mtype == "simulated",
            "real_inference": info["real_inference"],
            "note": _note_for(mtype),
        })
        print(f"  [generate] {name:<14} -> {avg * 1000:>8.1f} ms  [{mtype}]  answer={len(answer)} chars")

    return rows


# ---------------------------------------------------------------------------
# Real benchmark: end-to-end RAG
# ---------------------------------------------------------------------------

def benchmark_e2e_rag(
    doc_counts: List[int],
    backends: Dict[str, Dict[str, Any]],
    repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Benchmark full RAG pipeline: ingest + query + generate."""
    query = "什么是异构计算？它的优势是什么？"
    rows = []

    for count in doc_counts:
        for name, info in backends.items():
            if _skip_backend(name, info):
                continue
            backend = info["backend"]
            mtype = classify_backend(name, info)

            if not hasattr(backend, 'generate_answer'):
                continue

            total = 0.0
            for _ in range(repeats):
                # Reset
                if hasattr(backend, 'reset_corpus'):
                    backend.reset_corpus()

                t0 = time.perf_counter()

                # Ingest
                docs = generate_test_documents(count, size=300)
                all_chunks = chunk_texts(docs)
                if hasattr(backend, 'fit_and_embed'):
                    vectors = backend.fit_and_embed(all_chunks)
                else:
                    vectors = backend.embed_texts(all_chunks)

                # Query
                if hasattr(backend, 'transform'):
                    qvec = backend.transform([query])
                else:
                    qvec = backend.embed_texts([query])

                # Simple retrieval: cosine similarity
                best_idx = 0
                best_score = -1
                for i, vec in enumerate(vectors):
                    dot = sum(a * b for a, b in zip(qvec[0], vec))
                    if dot > best_score:
                        best_score = dot
                        best_idx = i

                # Generate
                context = all_chunks[best_idx] if best_idx < len(all_chunks) else ""
                answer = backend.generate_answer(query=query, context=context)

                total += time.perf_counter() - t0

            avg = total / repeats
            rows.append({
                "test": "end_to_end_rag",
                "doc_count": count,
                "chunk_count": len(all_chunks),
                "backend": name,
                "latency_s": round(avg, 6),
                "latency_ms": round(avg * 1000, 3),
                "measurement_type": mtype,
                "is_simulated": mtype == "simulated",
                "real_inference": info["real_inference"],
                "note": _note_for(mtype),
            })
            print(f"  [e2e_rag] docs={count:>3}, {name:<14} -> {avg * 1000:>8.1f} ms  [{mtype}]")

    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _note_for(mtype: str) -> str:
    return {
        "real_hardware": "Real hardware measurement",
        "cpu_fallback_with_hardware_detected": "Hardware EP detected but computation is CPU fallback (not real NPU/GPU inference)",
        "simulated": "Simulated via time.sleep multiplier, not real hardware",
        "unavailable": "Backend not available",
    }.get(mtype, mtype)


def collect_resource_usage() -> Dict[str, float]:
    if not HAS_PSUTIL:
        return {"cpu_percent": -1.0, "memory_percent": -1.0, "memory_used_mb": -1.0}
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    return {"cpu_percent": cpu, "memory_percent": mem.percent, "memory_used_mb": mem.used / (1024 * 1024)}


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
    parser.add_argument("--doc-counts", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--chunk-counts", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("  Real Hardware Benchmark")
    print("=" * 60)

    backends = detect_backends()

    print(f"\n  Backends detected:")
    for name, info in backends.items():
        avail = "✅" if info["available"] else "❌"
        real = "real" if info["real_inference"] else "fallback/sim"
        print(f"    {name:<14} available={avail}  inference={real}")

    real_backends = {n: i for n, i in backends.items() if i["real_inference"]}
    has_real = len(real_backends) > 1  # More than just CPU

    if has_real:
        print(f"\n  ✅ Real hardware backends: {list(real_backends.keys())}")
        print("  Running real + simulated benchmarks for comparison.")
    else:
        print("\n  ⚠️ No real GPU/NPU inference available.")
        print("  All results will be SIMULATED or CPU-only.")

    print(f"\n  Doc counts: {args.doc_counts}")
    print(f"  Chunk counts: {args.chunk_counts}")
    print(f"  Repeats: {args.repeats}")
    print("=" * 60)

    # Run all benchmarks
    all_rows = []

    print("\n[1/4] Embedding benchmark ...")
    all_rows.extend(benchmark_embedding(args.doc_counts, backends, args.repeats))

    print("\n[2/4] Query embedding benchmark ...")
    all_rows.extend(benchmark_query(args.chunk_counts, backends, args.repeats))

    print("\n[3/4] Generation benchmark ...")
    all_rows.extend(benchmark_generation(backends, args.repeats))

    print("\n[4/4] End-to-end RAG benchmark ...")
    all_rows.extend(benchmark_e2e_rag(args.doc_counts, backends, args.repeats))

    # Save
    print("\n[save] Writing CSV ...")
    save_csv(all_rows, "latency_results.csv")

    # Backend summary
    summary = []
    for name in set(r["backend"] for r in all_rows):
        subset = [r for r in all_rows if r["backend"] == name]
        mtype = subset[0]["measurement_type"]
        avg = sum(r["latency_ms"] for r in subset) / len(subset)
        summary.append({
            "backend": name,
            "avg_latency_ms": round(avg, 3),
            "test_count": len(subset),
            "measurement_type": mtype,
            "is_simulated": mtype == "simulated",
            "real_inference": subset[0].get("real_inference", False),
            "note": _note_for(mtype),
        })
    save_csv(summary, "backend_results.csv")
    save_csv([collect_resource_usage()], "resource_usage.csv")

    # Print warnings
    print("\n" + "=" * 60)
    if not has_real:
        print("  ⚠️ All results are SIMULATED or CPU-only.")
        print("  To get real GPU/NPU data, run on AMD hardware.")
    for name, info in backends.items():
        if info["available"] and not info["real_inference"]:
            print(f"  ⚠️ {name}: EP detected but inference is CPU fallback.")
            print(f"     CSV marked as 'cpu_fallback_with_hardware_detected', NOT 'real_hardware'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
