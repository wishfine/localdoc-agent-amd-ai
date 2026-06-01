"""
LocalDoc Agent - 智能体主控模块

LocalDocAgent 是整个 RAG（检索增强生成）管线的主控类，
负责协调文档加载、分块、向量化、检索和生成的完整流程。

管线流程：
  文档加载 -> 文本分块 -> 向量化 -> 索引存储 -> 查询检索 -> 答案生成

支持异步操作（asyncio），方便与前端或其他异步框架集成。
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
from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class LocalDocAgent:
    """
    本地文档智能体 - RAG 管线的主控类。

    整合文档加载、分块、向量化、检索和答案生成的完整流程。

    Args:
        backend: 推理后端实例（可选），需同时支持 embed_texts 和
                 generate_answer 方法。如果不提供，各组件将使用回退方案。
        scheduler: 任务调度器（可选），用于管理计算任务的调度

    用法示例：
        agent = LocalDocAgent(backend=npu_backend)
        agent.ingest_document("docs/guide.md")
        result = agent.query("什么是异构计算?")
        print(result['answer'])
    """

    def __init__(self, backend=None, scheduler=None) -> None:
        self.backend = backend
        self.scheduler = scheduler

        # 初始化各组件
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        self.embedding_engine = EmbeddingEngine(backend=backend)
        self.retriever = DocumentRetriever(embedding_engine=self.embedding_engine)
        self.generator = AnswerGenerator(backend=backend)

        # 统计信息
        self._ingested_files: list[str] = []
        self._total_chunks: int = 0

        backend_name = type(backend).__name__ if backend else "None (使用回退方案)"
        logger.info(
            f"LocalDocAgent 初始化完成\n"
            f"  后端: {backend_name}\n"
            f"  调度器: {type(scheduler).__name__ if scheduler else 'None'}"
        )

    def ingest_document(self, file_path: str) -> int:
        """
        导入单个文档到知识库。

        完整流程：加载 -> 分块 -> 向量化 -> 存入索引

        Args:
            file_path: 文档文件路径

        Returns:
            生成的文档块数量

        Raises:
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path)
        logger.info(f"开始导入文档: {path.name}")

        # 步骤 1：加载文档
        t0 = time.time()
        doc = self.loader.load_file(file_path)
        t_load = time.time() - t0
        logger.info(f"  [加载] {t_load:.3f}s - {len(doc['content'])} 字符")

        # 步骤 2：文本分块
        t0 = time.time()
        chunks = self.chunker.chunk_text(
            doc["content"], source=doc["source"]
        )
        t_chunk = time.time() - t0
        logger.info(f"  [分块] {t_chunk:.3f}s - {len(chunks)} 个块")

        if not chunks:
            logger.warning(f"文档未产生任何文本块: {path.name}")
            return 0

        # 步骤 3：向量化
        t0 = time.time()
        embeddings = self.embedding_engine.embed_chunks(chunks)
        t_embed = time.time() - t0
        logger.info(f"  [向量化] {t_embed:.3f}s - 维度 {len(embeddings[0]) if embeddings else 0}")

        # 步骤 4：将向量关联到块并存入索引
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunks_with_embeddings.append({
                **chunk,
                "embedding": embedding,
            })

        self.retriever.add_documents(chunks_with_embeddings)

        # 更新统计
        self._ingested_files.append(doc["source"])
        self._total_chunks += len(chunks)

        total_time = t_load + t_chunk + t_embed
        logger.info(
            f"文档导入完成: {path.name} - "
            f"{len(chunks)} 个块, 总耗时 {total_time:.3f}s"
        )
        return len(chunks)

    def ingest_directory(self, dir_path: str) -> int:
        """
        批量导入目录下的所有文档。

        Args:
            dir_path: 目录路径

        Returns:
            总共生成的文档块数量

        Raises:
            FileNotFoundError: 目录不存在
        """
        logger.info(f"开始导入目录: {dir_path}")
        t_start = time.time()

        # 步骤 1：加载目录下所有文件
        documents = self.loader.load_directory(dir_path)
        logger.info(f"  发现 {len(documents)} 个可加载文件")

        # 步骤 2-4：对每个文档进行分块、向量化、索引
        total_chunks = 0
        for i, doc in enumerate(documents, 1):
            source_name = Path(doc["source"]).name
            logger.info(f"  处理 [{i}/{len(documents)}]: {source_name}")

            try:
                chunks = self.chunker.chunk_text(
                    doc["content"], source=doc["source"]
                )
                if not chunks:
                    continue

                embeddings = self.embedding_engine.embed_chunks(chunks)

                chunks_with_embeddings = [
                    {**chunk, "embedding": emb}
                    for chunk, emb in zip(chunks, embeddings)
                ]

                self.retriever.add_documents(chunks_with_embeddings)

                self._ingested_files.append(doc["source"])
                self._total_chunks += len(chunks)
                total_chunks += len(chunks)

            except Exception as e:
                logger.error(
                    f"处理文件失败 [{source_name}]: "
                    f"{type(e).__name__}: {e}"
                )
                continue

        elapsed = time.time() - t_start
        logger.info(
            f"目录导入完成: {dir_path} - "
            f"{total_chunks} 个块, 耗时 {elapsed:.2f}s"
        )
        return total_chunks

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        对知识库进行查询并生成回答。

        完整流程：查询向量化 -> 检索 -> 答案生成

        Args:
            question: 用户问题
            top_k: 检索返回的最相关文档块数量（默认 3）

        Returns:
            字典，包含:
            - 'answer' (str): 生成的回答
            - 'sources' (list): 引用的来源文档列表
            - 'latency' (float): 查询总耗时（秒）
            - 'retrieved_chunks' (int): 检索到的文档块数量
        """
        logger.info(f"查询: '{question}' (top_k={top_k})")
        t_start = time.time()

        # 步骤 1：检索相关文档块
        t0 = time.time()
        relevant_chunks = self.retriever.retrieve(question, top_k=top_k)
        t_retrieve = time.time() - t0
        logger.info(f"  [检索] {t_retrieve:.3f}s - {len(relevant_chunks)} 个结果")

        # 步骤 2：生成回答
        t0 = time.time()
        answer = self.generator.generate(question, relevant_chunks)
        t_generate = time.time() - t0
        logger.info(f"  [生成] {t_generate:.3f}s")

        # 提取来源信息（去重）
        sources = list({
            chunk.get("source", "未知来源")
            for chunk in relevant_chunks
        })

        total_latency = time.time() - t_start

        result = {
            "answer": answer,
            "sources": sources,
            "latency": round(total_latency, 4),
            "retrieved_chunks": len(relevant_chunks),
        }

        logger.info(
            f"查询完成: 耗时 {total_latency:.3f}s, "
            f"来源数 {len(sources)}"
        )
        return result

    def get_stats(self) -> dict:
        """
        获取智能体的统计信息。

        Returns:
            字典，包含:
            - 'document_count' (int): 已导入的文档数量
            - 'chunk_count' (int): 索引中的文档块总数
            - 'backend' (str): 当前使用的后端名称
            - 'ingested_files' (list): 已导入的文件路径列表
        """
        backend_name = "None (回退模式)"
        if self.backend is not None:
            backend_name = type(self.backend).__name__

        return {
            "document_count": len(self._ingested_files),
            "chunk_count": self.retriever.get_document_count(),
            "backend": backend_name,
            "ingested_files": list(self._ingested_files),
        }

    # ============================================================
    # 异步接口（asyncio）
    # ============================================================

    async def aingest_document(self, file_path: str) -> int:
        """
        异步导入单个文档。

        将同步的 ingest_document 放入事件循环的线程池中执行，
        避免阻塞异步调用方。

        Args:
            file_path: 文档文件路径

        Returns:
            生成的文档块数量
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.ingest_document, file_path
        )

    async def aingest_directory(self, dir_path: str) -> int:
        """
        异步批量导入目录。

        Args:
            dir_path: 目录路径

        Returns:
            总共生成的文档块数量
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.ingest_directory, dir_path
        )

    async def aquery(self, question: str, top_k: int = 3) -> dict:
        """
        异步查询接口。

        Args:
            question: 用户问题
            top_k: 返回的最相关文档块数量

        Returns:
            与 query() 相同的结果字典
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.query, question, top_k
        )
