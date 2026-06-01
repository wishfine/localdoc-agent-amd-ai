"""
Test local LLM backend with a sample query.

Run: python scripts/test_llm.py
"""

import time
from pathlib import Path

from localdoc.backends.local_llm_backend import LocalLLMBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen3-1.7b"


def main():
    print("=" * 60)
    print("LocalDoc Agent - Test Local LLM")
    print(f"Model: Qwen3-1.7B")
    print(f"Model path: {MODEL_DIR}")
    print("=" * 60)

    backend = LocalLLMBackend(
        model_path=str(MODEL_DIR),
        max_new_tokens=128,
        context_chars=1600,
    )

    info = backend.get_device_info()
    print(f"后端名称: {backend.name}")
    print(f"模型本地可用: {backend.is_available()}")
    print(f"推理设备: {info['device']}")
    print(f"torch.cuda.is_available(): {info['torch_cuda_available']}")
    print(f"torch.version.hip: {info['torch_hip_version']}")
    print(f"硬件说明: {info['hardware_note']}")

    query = "请用三句话解释什么是异构计算。"
    context = (
        "异构计算是指在同一个计算系统中同时使用不同类型的处理单元，例如 CPU、GPU、NPU 或 FPGA。"
        "CPU 适合复杂控制逻辑和通用任务，GPU 适合大规模并行计算，NPU 适合低功耗神经网络推理。"
        "合理的异构调度能够根据任务特点选择合适的硬件，从而提高性能和能效。"
        "AMD 锐龙 AI MAX+ 处理器集成了 CPU、GPU 和 NPU 三种计算单元，是异构计算的典型代表。"
    )

    print(f"\n问题: {query}")
    print("\n正在生成回答 ...")

    start = time.perf_counter()
    answer = backend.generate_answer(query=query, context=context)
    elapsed = time.perf_counter() - start

    print("\n回答:")
    print("-" * 40)
    print(answer)
    print("-" * 40)
    print(f"\n耗时: {elapsed:.2f}s")
    print(f"设备: {info['device']}")

    if info["device"] == "cpu":
        print("\n⚠️ 当前在 CPU 上运行 Qwen3-1.7B 推理，不是 GPU/NPU 实测。")
    if not info["torch_hip_version"]:
        print("⚠️ 未检测到 AMD ROCm (torch.version.hip 为空)，不代表 AMD GPU 实测。")


if __name__ == "__main__":
    main()
