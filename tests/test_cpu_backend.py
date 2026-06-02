"""Tests for CPUBackend class."""

import pytest
from localdoc.backends.cpu_backend import CPUBackend


def test_fit_and_embed_returns_vectors():
    backend = CPUBackend()
    vectors = backend.fit_and_embed(["hello world", "test document"])
    assert len(vectors) == 2
    assert all(len(v) > 0 for v in vectors)


def test_transform_uses_existing_vocab():
    backend = CPUBackend()
    backend.fit_and_embed(["苹果 香蕉"])
    corpus_before = len(backend.get_corpus_texts())
    vecs = backend.transform(["火箭 卫星"])
    corpus_after = len(backend.get_corpus_texts())
    assert len(vecs) == 1
    assert corpus_before == corpus_after


def test_transform_dimension_matches_fit():
    backend = CPUBackend()
    doc_vecs = backend.fit_and_embed(["苹果 香蕉 葡萄"])
    query_vecs = backend.transform(["苹果"])
    assert len(doc_vecs[0]) == len(query_vecs[0])


def test_embed_texts_auto_fits():
    backend = CPUBackend()
    vectors = backend.embed_texts(["test"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


def test_embed_texts_after_fit_transforms():
    backend = CPUBackend()
    backend.fit_and_embed(["苹果 香蕉"])
    vecs = backend.embed_texts(["苹果"])
    assert len(vecs) == 1


def test_single_call_consistency():
    backend = CPUBackend()
    vecs = backend.fit_and_embed(["A B", "C D", "E F"])
    dims = [len(v) for v in vecs]
    assert len(set(dims)) == 1


def test_embed_texts_after_fit_preserves_vocab():
    """embed_texts after fit uses transform (doesn't rebuild vocab)."""
    backend = CPUBackend()
    backend.fit_and_embed(["苹果 香蕉"])
    dim1 = len(backend._vocab)
    backend.embed_texts(["火箭 卫星 导弹"])
    dim2 = len(backend._vocab)
    # embed_texts after fit calls transform — vocab unchanged
    assert dim2 == dim1


def test_generate_answer_string():
    backend = CPUBackend()
    answer = backend.generate_answer("什么是异构计算", "异构计算是多处理器架构。")
    assert len(answer) > 1


def test_generate_answer_list():
    backend = CPUBackend()
    answer = backend.generate_answer("test", ["文档一。", "文档二。"])
    assert len(answer) > 1


def test_generate_answer_empty():
    backend = CPUBackend()
    answer = backend.generate_answer("test", "")
    assert "未找到" in answer or "无法" in answer


def test_l2_normalization():
    backend = CPUBackend()
    vectors = backend.fit_and_embed(["hello world test"])
    norm = sum(v * v for v in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_benchmark_embedding():
    backend = CPUBackend()
    result = backend.benchmark_embedding(["text one", "text two"])
    assert result["backend"] == "CPU"
    assert result["embedding_dim"] > 0


def test_reset_corpus():
    backend = CPUBackend()
    backend.fit_and_embed(["苹果 香蕉", "葡萄 西瓜"])
    assert len(backend.get_corpus_texts()) == 2
    backend.reset_corpus()
    assert len(backend.get_corpus_texts()) == 0
    assert backend._vocab is None
    vecs = backend.fit_and_embed(["火箭"])
    assert len(vecs) == 1
