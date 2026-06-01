"""Tests for TextChunker class."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from localdoc.chunker import TextChunker


def test_basic_chunking(sample_text):
    """Test basic text chunking works."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_text(sample_text, source="test.txt")

    assert len(chunks) > 0
    # Each chunk should be a dict with expected keys
    for chunk in chunks:
        assert "content" in chunk
        assert "source" in chunk
        assert "index" in chunk
        assert "start_pos" in chunk


def test_chunk_size(sample_text):
    """Test chunks respect max size."""
    max_size = 100
    chunker = TextChunker(chunk_size=max_size, chunk_overlap=10)
    chunks = chunker.chunk_text(sample_text, source="test.txt")

    for chunk in chunks:
        # With overlap, content may slightly exceed chunk_size by up to overlap
        # but the base content should be within reason
        assert len(chunk["content"]) <= max_size + chunker.chunk_overlap + 10


def test_chunk_overlap(sample_text):
    """Test overlap works correctly."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=30)
    chunks = chunker.chunk_text(sample_text, source="test.txt")

    if len(chunks) > 1:
        # Overlap means second chunk's content should start with some of the
        # previous chunk's ending text (when overlap > 0)
        for i in range(1, len(chunks)):
            prev_content = chunks[i - 1]["content"]
            curr_content = chunks[i]["content"]
            # Check that there's at least some overlap
            overlap_text = prev_content[-chunker.chunk_overlap:]
            if len(overlap_text) > 0:
                # The current chunk should start with the overlap text
                # (or a substring of it if truncation happened)
                assert curr_content[: min(10, len(overlap_text))] in prev_content


def test_empty_text():
    """Test handles empty input."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text("", source="empty.txt")

    assert len(chunks) == 0


def test_whitespace_only_text():
    """Test handles whitespace-only input."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text("   \n\n   ", source="whitespace.txt")

    assert len(chunks) == 0


def test_long_paragraph():
    """Test splits long paragraphs."""
    long_text = "这是一个很长的段落。" * 100
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_text(long_text, source="long.txt")

    assert len(chunks) > 1


def test_markdown_content():
    """Test handles markdown headers/formatting."""
    markdown_text = """# 主标题

## 第一节

这是第一节的内容，包含一些详细说明。

## 第二节

这是第二节的内容。

- 列表项一
- 列表项二

**粗体文本** 和 *斜体文本*
"""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(markdown_text, source="doc.md")

    assert len(chunks) > 0
    # Should preserve some markdown structure
    all_content = " ".join(c["content"] for c in chunks)
    assert "主标题" in all_content or "第一节" in all_content


def test_source_metadata(sample_text):
    """Test chunks carry source info."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    source = "document.md"
    chunks = chunker.chunk_text(sample_text, source=source)

    for chunk in chunks:
        assert chunk["source"] == source
        assert isinstance(chunk["index"], int)
        assert isinstance(chunk["start_pos"], int)


def test_chunk_documents():
    """Test batch chunking of multiple documents."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    documents = [
        {"content": "第一篇文档的内容。这是关于异构计算的介绍。", "source": "doc1.md"},
        {"content": "第二篇文档的内容。这是关于 NPU 的说明。", "source": "doc2.md"},
    ]
    all_chunks = chunker.chunk_documents(documents)

    assert len(all_chunks) > 0
    sources = {c["source"] for c in all_chunks}
    assert "doc1.md" in sources
    assert "doc2.md" in sources


def test_invalid_chunk_size():
    """Test raises error for invalid chunk_size."""
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, chunk_overlap=10)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=-1, chunk_overlap=10)


def test_invalid_overlap():
    """Test raises error for invalid overlap."""
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=-1)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
