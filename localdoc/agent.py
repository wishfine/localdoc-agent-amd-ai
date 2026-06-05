"""
LocalDoc Agent - 智能体主控模块

LocalDocAgent 是 RAG 管线的主控类。
当提供了 scheduler 时，每个阶段通过调度器选择后端并记录调度日志。
当 scheduler 为 None 时，直接执行各阶段。

管线流程：
  文档加载 -> 文本分块 -> 向量化 -> 索引存储 -> 查询检索 -> 答案生成
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

from localdoc.loader import DocumentLoader
from localdoc.chunker import TextChunker
from localdoc.embedding import EmbeddingEngine
from localdoc.retriever import DocumentRetriever
from localdoc.generator import AnswerGenerator
from localdoc.scheduler import BenchmarkTaskType
from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class LocalDocAgent:
    """
    本地文档智能体 - RAG 管线的主控类。

    跟踪所有已摄入的 chunk，每次新增文档后重新构建向量索引，
    保证所有文档的向量维度一致。
    """

    def __init__(self, backend=None, scheduler=None) -> None:
        self.backend = backend
        self.scheduler = scheduler

        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        self.embedding_engine = EmbeddingEngine(backend=backend)
        self.retriever = DocumentRetriever(embedding_engine=self.embedding_engine)
        self.generator = AnswerGenerator(backend=backend)

        # 跟踪所有已摄入的 chunk 和文件
        self._all_chunks: list = []
        self._ingested_files: list[str] = []

        backend_name = type(backend).__name__ if backend else "None (使用回退方案)"
        logger.info(
            f"LocalDocAgent 初始化完成\n"
            f"  后端: {backend_name}\n"
            f"  调度器: {'已启用' if scheduler else '未启用'}"
        )

    def _rebuild_index(self) -> None:
        """
        用当前所有 chunk 重新构建向量索引。

        保证所有文档的向量维度一致（基于同一个词汇表）。
        """
        if not self._all_chunks:
            return

        # 重置后端和引擎状态，确保从零构建词汇表
        if self.backend and hasattr(self.backend, 'reset_corpus'):
            self.backend.reset_corpus()
        self.embedding_engine.reset()

        # 用全部 chunk 构建词汇表和向量。调度器在这里记录 embedding
        # 阶段的策略选择；实际计算由 embedding_engine 及其后端完成。
        if self.scheduler:
            embeddings = self.scheduler.execute(
                BenchmarkTaskType.EMBEDDING,
                self.embedding_engine.embed_chunks,
                self._all_chunks,
            )
        else:
            embeddings = self.embedding_engine.embed_chunks(self._all_chunks)

        # 重建 retriever 索引
        self.retriever.clear()
        chunks_with_embeddings = [
            {**chunk, "embedding": emb}
            for chunk, emb in zip(self._all_chunks, embeddings)
        ]
        self.retriever.add_documents(chunks_with_embeddings)

        logger.info("向量索引重建完成: %d 个块", len(self._all_chunks))

    def ingest_document(self, file_path: str) -> int:
        """
        导入单个文档到知识库。

        流程：加载 -> 切块 -> 追加到全局 chunk 列表 -> 重建向量索引

        Args:
            file_path: 文档文件路径

        Returns:
            生成的文档块数量
        """
        path = Path(file_path)
        logger.info(f"开始导入文档: {path.name}")

        if self.scheduler:
            doc = self.scheduler.execute(
                BenchmarkTaskType.DOCUMENT_LOADING, self.loader.load_file, file_path
            )
            chunks = self.scheduler.execute(
                BenchmarkTaskType.CHUNKING,
                lambda: self.chunker.chunk_text(doc["content"], source=doc["source"]),
            )
        else:
            doc = self.loader.load_file(file_path)
            chunks = self.chunker.chunk_text(doc["content"], source=doc["source"])

        if not chunks:
            logger.warning(f"文档未产生任何文本块: {path.name}")
            return 0

        # 追加到全局 chunk 列表
        self._all_chunks.extend(chunks)
        self._ingested_files.append(doc["source"])

        # 重建向量索引（所有 chunk 统一词汇表）
        self._rebuild_index()

        logger.info(f"文档导入完成: {path.name} - {len(chunks)} 个块，总块数 {len(self._all_chunks)}")
        return len(chunks)

    def ingest_directory(self, dir_path: str) -> int:
        """批量导入目录下的所有文档。"""
        logger.info(f"开始导入目录: {dir_path}")
        t_start = time.time()

        if self.scheduler:
            documents = self.scheduler.execute(
                BenchmarkTaskType.DOCUMENT_LOADING,
                self.loader.load_directory,
                dir_path,
            )
        else:
            documents = self.loader.load_directory(dir_path)
        logger.info(f"  发现 {len(documents)} 个可加载文件")

        for doc in documents:
            try:
                if self.scheduler:
                    chunks = self.scheduler.execute(
                        BenchmarkTaskType.CHUNKING,
                        self.chunker.chunk_text,
                        doc["content"],
                        source=doc["source"],
                    )
                else:
                    chunks = self.chunker.chunk_text(doc["content"], source=doc["source"])
                self._all_chunks.extend(chunks)
                self._ingested_files.append(doc["source"])
            except Exception as e:
                logger.error(f"处理文件失败 [{Path(doc['source']).name}]: {e}")

        # 统一重建索引
        self._rebuild_index()

        elapsed = time.time() - t_start
        logger.info(f"目录导入完成: {dir_path} - {len(self._all_chunks)} 个块, 耗时 {elapsed:.2f}s")
        return len(self._all_chunks)

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        对知识库进行查询并生成回答。

        Returns:
            字典: answer, sources, latency, retrieved_chunks, backend_trace
        """
        logger.info(f"查询: '{question}' (top_k={top_k})")
        t_start = time.time()
        scheduling_log = []

        if self.scheduler:
            relevant_chunks = self.scheduler.execute(
                BenchmarkTaskType.RETRIEVAL, self.retriever.retrieve, question, top_k
            )
            scheduling_log.append(self.scheduler.get_execution_log()[-1])

            answer = self.scheduler.execute(
                BenchmarkTaskType.GENERATION, self.generator.generate, question, relevant_chunks
            )
            scheduling_log.append(self.scheduler.get_execution_log()[-1])
        else:
            relevant_chunks = self.retriever.retrieve(question, top_k=top_k)
            answer = self.generator.generate(question, relevant_chunks)

        sources = list({
            chunk.get("source", "未知来源") for chunk in relevant_chunks
        })

        total_latency = time.time() - t_start

        result = {
            "answer": answer,
            "sources": sources,
            "latency": round(total_latency, 4),
            "retrieved_chunks": len(relevant_chunks),
        }

        if self.scheduler and scheduling_log:
            result["backend_trace"] = [
                {
                    "task_type": entry["task_type"],
                    "backend": entry["backend"],
                    "reason": entry["reason"],
                    "elapsed_seconds": entry["elapsed_seconds"],
                    "is_simulated": entry.get("is_simulated", False),
                }
                for entry in scheduling_log
            ]

        logger.info(f"查询完成: {total_latency:.3f}s, 来源数 {len(sources)}")
        return result

    def get_stats(self) -> dict:
        """获取智能体的统计信息。"""
        backend_name = type(self.backend).__name__ if self.backend else "None (回退模式)"
        return {
            "document_count": len(self._ingested_files),
            "chunk_count": len(self._all_chunks),
            "backend": backend_name,
            "ingested_files": list(self._ingested_files),
            "scheduler_enabled": self.scheduler is not None,
            "available_backends": list(self.scheduler.backends.keys()) if self.scheduler else [],
        }

    # ============================================================
    # 异步接口
    # ============================================================

    async def aingest_document(self, file_path: str) -> int:
        return await asyncio.to_thread(self.ingest_document, file_path)

    async def aingest_directory(self, dir_path: str) -> int:
        return await asyncio.to_thread(self.ingest_directory, dir_path)

    async def aquery(self, question: str, top_k: int = 3) -> dict:
        return await asyncio.to_thread(self.query, question, top_k)
