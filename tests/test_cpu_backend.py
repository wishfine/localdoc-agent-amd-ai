"""Tests for CPUBackend class — regression tests for P0-1 and vocab persistence."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from localdoc.backends.cpu_backend import CPUBackend


def test_embed_texts_returns_vectors():
    """Basic: embed_texts returns a list of lists."""
    backend = CPUBackend()
    vectors = backend.embed_texts(["hello world", "test document"])
    assert len(vectors) == 2
    assert all(isinstance(v, list) for v in vectors)
    assert all(len(v) > 0 for v in vectors)


def test_embed_texts_dimension_consistency():
    """P0-1 regression: multiple calls must produce same-dimension vectors."""
    backend = CPUBackend()
    # First call: ingest documents
    vecs1 = backend.embed_texts(["异构计算是指使用不同类型处理器", "CPU 适合通用任务"])
    dim1 = len(vecs1[0])
    assert dim1 > 0

    # Second call: query (may introduce new words)
    vecs2 = backend.embed_texts(["什么是异构计算"])
    dim2 = len(vecs2[0])

    # Dimensions must be consistent (vocab only grows, never shrinks)
    assert dim2 >= dim1, f"Dimension shrunk: {dim1} -> {dim2}"


def test_embed_texts_no_type_error():
    """P0-1 regression: embed_texts must not throw TypeError on len()."""
    backend = CPUBackend()
    # This used to crash with: TypeError: object of type 'int' has no len()
    vectors = backend.embed_texts(["test"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


def test_embed_texts_empty():
    """Edge case: empty input."""
    backend = CPUBackend()
    vectors = backend.embed_texts([])
    assert vectors == []


def test_embed_texts_single_text():
    """Edge case: single text."""
    backend = CPUBackend()
    vectors = backend.embed_texts(["hello"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


def test_embed_texts_vocab_grows():
    """Vocab must grow when new words appear."""
    backend = CPUBackend()
    vecs1 = backend.embed_texts(["苹果 香蕉"])
    dim1 = len(vecs1[0])

    vecs2 = backend.embed_texts(["葡萄 西瓜"])
    dim2 = len(vecs2[0])

    # New words should expand the vocab
    assert dim2 > dim1, f"Vocab didn't grow: {dim1} -> {dim2}"


def test_generate_answer_with_string_context():
    """generate_answer must accept string context (not just List[str])."""
    backend = CPUBackend()
    context = "异构计算是指使用不同类型处理器的计算架构。CPU 适合通用任务。"
    answer = backend.generate_answer(query="什么是异构计算", context=context)
    assert len(answer) > 1
    assert isinstance(answer, str)


def test_generate_answer_with_list_context():
    """generate_answer must also accept List[str]."""
    backend = CPUBackend()
    context = ["异构计算是指使用不同类型处理器的计算架构。", "CPU 适合通用任务。"]
    answer = backend.generate_answer(query="什么是异构计算", context=context)
    assert len(answer) > 1


def test_generate_answer_empty_context():
    """Edge case: empty context."""
    backend = CPUBackend()
    answer = backend.generate_answer(query="test", context="")
    assert "未找到" in answer or "无法" in answer


def test_l2_normalization():
    """Vectors must be L2 normalized."""
    backend = CPUBackend()
    vectors = backend.embed_texts(["hello world test"])
    vec = vectors[0]
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6, f"Vector not normalized: norm={norm}"


def test_benchmark_embedding():
    """benchmark_embedding returns expected fields."""
    backend = CPUBackend()
    result = backend.benchmark_embedding(["test text one", "test text two"])
    assert "backend" in result
    assert result["backend"] == "CPU"
    assert "elapsed_seconds" in result
    assert "embedding_dim" in result
    assert result["embedding_dim"] > 0
