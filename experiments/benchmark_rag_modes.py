"""
RAG Mode Comparison Benchmark

Compare extractive answer vs Qwen3-1.7B local LLM answer.
All results marked as local LLM inference, NOT AMD hardware benchmark.

Output:
- results/rag_mode_comparison.csv
- results/rag_stage_breakdown.csv
"""

import csv
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen3-1.7b"

TEST_QUERY = "什么是异构计算？它有什么优势？"

TEST_DOCUMENT = """# 异构计算概述

异构计算是指在同一个计算系统中同时使用不同类型的处理单元来执行不同类型的任务。

## 主要计算单元

CPU 适合复杂控制逻辑和串行任务。GPU 适合大规模并行浮点计算。NPU 专为神经网络推理优化，功耗低、效率高。

## 异构调度

合理的异构调度策略能够根据任务的计算特征，将任务分配到最合适的硬件上，从而最大化系统整体性能和能效。

## AMD 锐龙 AI MAX+

AMD 锐龙 AI MAX+ 处理器集成了 CPU、RDNA 3.5 架构 iGPU 和 XDNA 2 架构 NPU，是端侧异构计算的典型代表。
"""


def check_llm_available() -> bool:
    return MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()


def run_extractive_mode():
    """Run RAG pipeline with extractive answer generation."""
    from localdoc.agent import LocalDocAgent

    agent = LocalDocAgent()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(TEST_DOCUMENT)
        tmpfile = f.name

    t0 = time.perf_counter()
    n = agent.ingest_document(tmpfile)
    ingest_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = agent.query(TEST_QUERY, top_k=3)
    query_time = time.perf_counter() - t0

    Path(tmpfile).unlink(missing_ok=True)

    return {
        "mode": "extractive",
        "ingest_time_s": round(ingest_time, 4),
        "query_time_s": round(query_time, 4),
        "total_time_s": round(ingest_time + query_time, 4),
        "answer_length": len(result["answer"]),
        "answer_preview": result["answer"][:80],
        "chunks_ingested": n,
    }


def run_llm_mode():
    """Run RAG pipeline with local LLM answer generation."""
    from localdoc.agent import LocalDocAgent
    from localdoc.backends.local_llm_backend import LocalLLMBackend

    backend = LocalLLMBackend(
        model_path=str(MODEL_DIR),
        max_new_tokens=128,
        context_chars=1600,
    )
    agent = LocalDocAgent(backend=backend)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(TEST_DOCUMENT)
        tmpfile = f.name

    t0 = time.perf_counter()
    n = agent.ingest_document(tmpfile)
    ingest_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = agent.query(TEST_QUERY, top_k=3)
    query_time = time.perf_counter() - t0

    Path(tmpfile).unlink(missing_ok=True)

    info = backend.get_device_info()
    return {
        "mode": "local_llm_qwen3",
        "ingest_time_s": round(ingest_time, 4),
        "query_time_s": round(query_time, 4),
        "total_time_s": round(ingest_time + query_time, 4),
        "answer_length": len(result["answer"]),
        "answer_preview": result["answer"][:80],
        "chunks_ingested": n,
        "device": info["device"],
        "torch_hip_version": info["torch_hip_version"] or "N/A",
        "is_amd_hardware_benchmark": info["is_amd_hardware_benchmark"],
    }


def main():
    print("=" * 60)
    print("  RAG Mode Comparison Benchmark")
    print("  ⚠️ Local LLM inference, NOT AMD hardware benchmark")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Always run extractive
    print("\n[1] 运行抽取式回答 (extractive) ...")
    extractive_result = run_extractive_mode()
    print(f"  耗时: {extractive_result['total_time_s']:.3f}s")
    print(f"  回答: {extractive_result['answer_preview']}...")

    # Run LLM if available
    llm_result = None
    if check_llm_available():
        print("\n[2] 运行本地 LLM 回答 (Qwen3-1.7B) ...")
        try:
            llm_result = run_llm_mode()
            print(f"  耗时: {llm_result['total_time_s']:.3f}s")
            print(f"  设备: {llm_result['device']}")
            print(f"  回答: {llm_result['answer_preview']}...")
        except Exception as e:
            print(f"  ❌ LLM 运行失败: {e}")
            llm_result = None
    else:
        print(f"\n[2] 模型未找到: {MODEL_DIR}")
        print("  跳过 local_llm_qwen3 模式。")
        print("  运行 bash scripts/download_llm.sh 下载模型。")

    # Save comparison CSV
    csv_path = RESULTS_DIR / "rag_mode_comparison.csv"
    rows = []
    base_fields = {
        "model_name": "N/A",
        "model_id": "N/A",
        "device": "cpu",
        "torch_cuda_available": False,
        "torch_hip_version": "N/A",
        "is_local_llm": False,
        "is_amd_hardware_benchmark": False,
        "note": "Extractive mode; not LLM. Not AMD hardware benchmark.",
    }
    rows.append({
        "mode": extractive_result["mode"],
        "ingest_time_s": extractive_result["ingest_time_s"],
        "query_time_s": extractive_result["query_time_s"],
        "total_time_s": extractive_result["total_time_s"],
        "answer_length": extractive_result["answer_length"],
        "chunks_ingested": extractive_result["chunks_ingested"],
        **base_fields,
    })

    if llm_result:
        rows.append({
            "mode": llm_result["mode"],
            "ingest_time_s": llm_result["ingest_time_s"],
            "query_time_s": llm_result["query_time_s"],
            "total_time_s": llm_result["total_time_s"],
            "answer_length": llm_result["answer_length"],
            "chunks_ingested": llm_result["chunks_ingested"],
            "model_name": "Qwen3-1.7B",
            "model_id": "Qwen/Qwen3-1.7B",
            "device": llm_result["device"],
            "torch_cuda_available": llm_result.get("torch_cuda_available", False),
            "torch_hip_version": llm_result.get("torch_hip_version", "N/A"),
            "is_local_llm": True,
            "is_amd_hardware_benchmark": llm_result.get("is_amd_hardware_benchmark", False),
            "note": "Local LLM inference only; not AMD GPU/NPU hardware benchmark.",
        })
    else:
        rows.append({
            "mode": "local_llm_qwen3",
            "ingest_time_s": "SKIPPED",
            "query_time_s": "SKIPPED",
            "total_time_s": "SKIPPED",
            "answer_length": "SKIPPED",
            "chunks_ingested": "SKIPPED",
            "model_name": "Qwen3-1.7B",
            "model_id": "Qwen/Qwen3-1.7B",
            "device": "N/A",
            "torch_cuda_available": "N/A",
            "torch_hip_version": "N/A",
            "is_local_llm": True,
            "is_amd_hardware_benchmark": False,
            "note": "Model not found. Skipped. Not AMD hardware benchmark.",
        })

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ 对比结果: {csv_path}")

    # Save stage breakdown CSV
    breakdown_path = RESULTS_DIR / "rag_stage_breakdown.csv"
    breakdown_rows = []
    for r in rows:
        mode = r["mode"]
        if r["ingest_time_s"] == "SKIPPED":
            breakdown_rows.append({
                "mode": mode, "stage": "ingest", "time_s": "SKIPPED",
                "is_local_llm": r["is_local_llm"],
                "note": r["note"],
            })
            breakdown_rows.append({
                "mode": mode, "stage": "query", "time_s": "SKIPPED",
                "is_local_llm": r["is_local_llm"],
                "note": r["note"],
            })
        else:
            breakdown_rows.append({
                "mode": mode, "stage": "ingest",
                "time_s": r["ingest_time_s"],
                "is_local_llm": r["is_local_llm"],
                "note": r["note"],
            })
            breakdown_rows.append({
                "mode": mode, "stage": "query",
                "time_s": r["query_time_s"],
                "is_local_llm": r["is_local_llm"],
                "note": r["note"],
            })

    bd_fieldnames = list(breakdown_rows[0].keys())
    with open(breakdown_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bd_fieldnames)
        writer.writeheader()
        writer.writerows(breakdown_rows)
    print(f"✅ 阶段分解: {breakdown_path}")


if __name__ == "__main__":
    main()
