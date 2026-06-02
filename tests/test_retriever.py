"""Tests for DocumentRetriever class."""

import pytest
from localdoc.embedding import EmbeddingEngine
from localdoc.retriever import DocumentRetriever


@pytest.fixture
def retriever_with_docs(sample_chunks):
    """Create a retriever populated with sample chunks and their embeddings."""
    engine = EmbeddingEngine(backend=None)  # TF-IDF mode
    # Compute embeddings for the chunks
    embeddings = engine.embed_chunks(sample_chunks)

    chunks_with_embeddings = []
    for chunk, emb in zip(sample_chunks, embeddings):
        chunks_with_embeddings.append({**chunk, "embedding": emb})

    retriever = DocumentRetriever(embedding_engine=engine)
    retriever.add_documents(chunks_with_embeddings)
    return retriever


def test_add_and_retrieve(retriever_with_docs):
    """Test add docs and retrieve relevant ones."""
    results = retriever_with_docs.retrieve("异构计算 CPU GPU")

    assert len(results) > 0
    for r in results:
        assert "content" in r
        assert "score" in r
        assert "source" in r


def test_top_k(retriever_with_docs):
    """Test respects top_k parameter."""
    top_k = 2
    results = retriever_with_docs.retrieve("异构计算处理器", top_k=top_k)
    assert len(results) <= top_k


def test_empty_index():
    """Test handles empty document store."""
    engine = EmbeddingEngine(backend=None)
    retriever = DocumentRetriever(embedding_engine=engine)
    results = retriever.retrieve("any query")

    assert len(results) == 0


def test_relevance_order(retriever_with_docs):
    """Test results are ordered by relevance."""
    results = retriever_with_docs.retrieve("CPU GPU NPU 调度")

    if len(results) > 1:
        # Results should be ordered by score (descending)
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]


def test_query_no_match(retriever_with_docs):
    """Test handles queries with no good match."""
    # Query with content very different from stored docs
    results = retriever_with_docs.retrieve(
        "quantum entanglement dark matter singularity"
    )

    # Should still return results (TF-IDF won't crash), but scores may be low
    assert isinstance(results, list)


def test_get_document_count(retriever_with_docs):
    """Test document count tracking."""
    assert retriever_with_docs.get_document_count() == 5


def test_clear(retriever_with_docs):
    """Test clearing the index."""
    retriever_with_docs.clear()
    assert retriever_with_docs.get_document_count() == 0
    results = retriever_with_docs.retrieve("test query")
    assert len(results) == 0
