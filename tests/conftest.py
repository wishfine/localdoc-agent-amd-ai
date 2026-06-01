"""Shared pytest fixtures for localdoc-agent-amd-ai tests."""

import sys
import os

# Ensure the project root is on the path so `import localdoc` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture
def sample_text():
    """Sample text for testing the chunker."""
    return (
        "异构计算是指在单一系统中使用不同类型处理器的计算架构。\n\n"
        "AMD 锐龙 AI MAX+ 处理器集成了 CPU、GPU 和 NPU 三种计算单元。"
        "CPU 负责通用计算任务，GPU 擅长并行浮点运算，NPU 则专为神经网络推理优化。\n\n"
        "在本地知识库智能体中，我们利用异构调度器将不同任务分配到最适合的硬件上。"
        "文档加载和文本分块由 CPU 处理，向量嵌入优先使用 NPU，"
        "答案生成优先使用 GPU。当特定硬件不可用时，系统自动回退到 CPU。\n\n"
        "这种调度策略能够充分发挥异构硬件的性能优势，同时保证系统在任何环境下都能正常运行。"
    )


@pytest.fixture
def sample_chunks():
    """Sample chunk dicts matching TextChunker.chunk_text() output format."""
    return [
        {
            "content": "异构计算是指在单一系统中使用不同类型处理器的计算架构。",
            "source": "test_intro.md",
            "index": 0,
            "start_pos": 0,
        },
        {
            "content": "AMD 锐龙 AI MAX+ 处理器集成了 CPU、GPU 和 NPU 三种计算单元。",
            "source": "test_intro.md",
            "index": 1,
            "start_pos": 50,
        },
        {
            "content": "文档加载和文本分块由 CPU 处理，向量嵌入优先使用 NPU。",
            "source": "test_design.md",
            "index": 0,
            "start_pos": 0,
        },
        {
            "content": "答案生成优先使用 GPU，当特定硬件不可用时回退到 CPU。",
            "source": "test_design.md",
            "index": 1,
            "start_pos": 60,
        },
        {
            "content": "这种调度策略能够充分发挥异构硬件的性能优势。",
            "source": "test_summary.md",
            "index": 0,
            "start_pos": 0,
        },
    ]


@pytest.fixture
def cpu_backend():
    """A CPUBackend instance."""
    from localdoc.backends.cpu_backend import CPUBackend

    return CPUBackend()
