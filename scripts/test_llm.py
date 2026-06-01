"""
Test local LLM backend with a sample query.

Run: python scripts/test_llm.py
"""

import time
from pathlib import Path

from localdoc.backends.local_llm_backend import LocalLLMBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct"


def main():
    print("=" * 60)
    print("LocalDoc Agent - Test Local LLM")
    print(f"Model path: {MODEL_DIR}")
    print("=" * 60)

    backend = LocalLLMBackend(
        model_path=str(MODEL_DIR),
        max_new_tokens=128,
        context_chars=1600,
    )

    print(f"后端名称: {backend.name}")
    print(f"模型可用: {backend.is_available()}")
    print(f"设备信息: {backend.get_device_info()}")

    query = "请用三句话解释什么是异构计算。"
    context = (
        "异构计算是指在同一个计算系统中同时使用不同类型的处理单元，例如 CPU、GPU、NPU 或 FPGA。"
        "CPU 适合复杂控制逻辑和通用任务，GPU 适合大规模并行计算，NPU 适合低功耗神经网络推理。"
        "合理的异构调度能够根据任务特点选择合适的硬件，从而提高性能和能效。"
        "AMD 锐龙 AI MAX+ 处理器集成了 CPU、GPU 和 NPU 三种计算单元，是异构计算的典型代表。"
    )

    print("\n问题:", query)
    print("\n正在生成回答 ...")

    start = time.perf_counter()
    answer = backend.generate_answer(query=query, context=context)
    elapsed = time.perf_counter() - start

    print("\n回答:")
    print("-" * 40)
    print(answer)
    print("-" * 40)
    print(f"\n耗时: {elapsed:.2f}s")
    print(f"设备: {backend.get_device_info()['device']}")

    if backend.get_device_info()["device"] == "cpu":
        print("\n⚠️ 当前在 CPU 上运行推理，不是 GPU/NPU 实测。")


if __name__ == "__main__":
    main()
