"""
LLM Generation Latency Benchmark

Benchmark Qwen3-1.7B local inference latency on short queries.
All results are marked as local LLM inference, NOT AMD hardware benchmark.

Output: results/llm_generation_benchmark.csv
"""

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR = PROJECT_ROOT / "models" / "qwen3-1.7b"

TEST_QUERIES = [
    "请用三句话解释什么是异构计算。",
    "请说明 CPU、GPU、NPU 分别适合什么任务。",
    "为什么本地知识库智能体适合隐私敏感场景？",
]

TEST_CONTEXT = (
    "异构计算是指在同一个计算系统中同时使用不同类型的处理单元，例如 CPU、GPU、NPU 或 FPGA。"
    "CPU 适合复杂控制逻辑和通用任务，GPU 适合大规模并行计算，NPU 适合低功耗神经网络推理。"
    "合理的异构调度能够根据任务特点选择合适的硬件，从而提高性能和能效。"
    "本地知识库智能体将文档存储和推理过程完全放在本地完成，不依赖云端服务，保护用户数据隐私。"
)


def check_model_exists() -> bool:
    return MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()


def get_memory_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().used / (1024 * 1024)
    except ImportError:
        return -1.0


def main():
    print("=" * 60)
    print("  LLM Generation Latency Benchmark")
    print("  Model: Qwen3-1.7B")
    print("  ⚠️ Local LLM inference, NOT AMD hardware benchmark")
    print("=" * 60)

    if not check_model_exists():
        print(f"\n❌ 模型未找到: {MODEL_DIR}")
        print("\n请先运行:")
        print("  bash scripts/setup_llm.sh")
        print("  bash scripts/download_llm.sh")
        print("\n跳过 LLM benchmark。")
        # Write empty CSV with headers
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / "llm_generation_benchmark.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_id", "query", "model_name", "model_id", "device",
                "torch_cuda_available", "torch_hip_version",
                "model_load_time_s", "generation_time_s",
                "input_chars", "output_chars", "input_tokens", "output_tokens",
                "tokens_per_second", "memory_before_mb", "memory_after_mb",
                "is_local_llm", "is_amd_hardware_benchmark", "note",
            ])
            writer.writerow([
                "SKIPPED", "Model not found", "Qwen3-1.7B", "Qwen/Qwen3-1.7B",
                "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                "N/A", "N/A", "N/A", True, False,
                "Model not found. Skipped. Not AMD hardware benchmark.",
            ])
        print(f"\n已写入: {csv_path}")
        return

    from localdoc.backends.local_llm_backend import LocalLLMBackend

    backend = LocalLLMBackend(
        model_path=str(MODEL_DIR),
        max_new_tokens=128,
        context_chars=1600,
    )

    rows = []
    for i, query in enumerate(TEST_QUERIES):
        print(f"\n[{i+1}/{len(TEST_QUERIES)}] {query[:40]}...")

        mem_before = get_memory_mb()

        # Measure generation time
        t0 = time.perf_counter()
        answer = backend.generate_answer(query=query, context=TEST_CONTEXT)
        gen_time = time.perf_counter() - t0

        mem_after = get_memory_mb()

        # Get device info AFTER model is loaded and inference is done
        info_after = backend.get_device_info()

        # Estimate token counts (rough: 1 Chinese char ≈ 2 tokens)
        input_chars = len(TEST_CONTEXT) + len(query)
        output_chars = len(answer)
        input_tokens_est = int(input_chars * 1.5)
        output_tokens_est = int(output_chars * 1.5)
        tps = output_tokens_est / gen_time if gen_time > 0 else 0

        row = {
            "query_id": i + 1,
            "query": query,
            "model_name": "Qwen3-1.7B",
            "model_id": "Qwen/Qwen3-1.7B",
            "device": info_after["device"],
            "torch_cuda_available": info_after["torch_cuda_available"],
            "torch_hip_version": info_after["torch_hip_version"] or "N/A",
            "model_load_time_s": info_after["load_time_s"],
            "generation_time_s": round(gen_time, 3),
            "input_chars": input_chars,
            "output_chars": output_chars,
            "input_tokens": input_tokens_est,
            "output_tokens": output_tokens_est,
            "tokens_per_second": round(tps, 1),
            "memory_before_mb": round(mem_before, 1),
            "memory_after_mb": round(mem_after, 1),
            "is_local_llm": True,
            "is_amd_hardware_benchmark": False,
            "is_rocm_runtime_detected": info_after.get("is_rocm_runtime_detected", False),
            "note": info_after["note"],
        }
        rows.append(row)

        print(f"  回答: {answer[:60]}...")
        print(f"  耗时: {gen_time:.2f}s, 输出: {output_chars} 字符, ~{tps:.1f} tokens/s")

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "llm_generation_benchmark.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ 结果已保存: {csv_path}")

    first_info = rows[0] if rows else {}
    if first_info.get("device") == "cpu":
        print("\n⚠️ 当前在 CPU 上运行 Qwen3-1.7B 推理，不是 GPU/NPU 实测。")


if __name__ == "__main__":
    main()
