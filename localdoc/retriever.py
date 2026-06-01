"""
LocalDoc Agent - 文档检索器模块

负责存储文档块及其向量，并根据查询进行语义检索。

工作流程：
1. add_documents: 接收带有向量的文档块，存入内存索引
2. retrieve: 对查询文本计算向量，与所有文档块计算余弦相似度，返回最相关的 top_k 个

数据全部保存在内存中，适合中小规模知识库（数千个文档块）。
"""

from typing import Optional

from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRetriever:
    """
    文档检索器 - 基于向量相似度的语义检索引擎。

    内部维护两个平行列表：
    - _documents: 存储文档块元数据（字典）
    - _embeddings: 存储对应的向量表示

    检索时将查询向量与所有文档向量计算余弦相似度，
    返回相似度最高的 top_k 个文档块。

    Args:
        embedding_engine: 向量化引擎实例，用于将查询转换为向量

    用法示例：
        retriever = DocumentRetriever(embedding_engine=engine)
        retriever.add_documents(chunks_with_embeddings)
        results = retriever.retrieve("什么是异构计算?", top_k=3)
    """

    def __init__(self, embedding_engine=None) -> None:
        self.embedding_engine = embedding_engine

        # 平行列表：_documents[i] 的向量为 _embeddings[i]
        self._documents: list[dict] = []
        self._embeddings: list[list] = []

        logger.info("DocumentRetriever 初始化完成")

    def add_documents(self, chunks_with_embeddings: list) -> None:
        """
        将带有向量的文档块添加到检索索引中。

        Args:
            chunks_with_embeddings: 文档块列表，每个元素为字典，需包含:
                - 'content' (str): 文本内容
                - 'embedding' (list): 向量表示
                - 'source' (str, 可选): 来源标识
                - 'index' (int, 可选): 块序号

        Raises:
            ValueError: 输入数据格式不正确
        """
        if not chunks_with_embeddings:
            logger.warning("输入的文档块列表为空")
            return

        added_count = 0
        for chunk in chunks_with_embeddings:
            # 验证必要字段
            if "content" not in chunk:
                logger.warning("跳过缺少 'content' 字段的文档块")
                continue
            if "embedding" not in chunk:
                logger.warning(
                    f"跳过缺少 'embedding' 字段的文档块: "
                    f"{chunk.get('source', 'unknown')}"
                )
                continue

            self._documents.append({
                "content": chunk["content"],
                "source": chunk.get("source", ""),
                "index": chunk.get("index", len(self._documents)),
                "start_pos": chunk.get("start_pos", -1),
            })
            self._embeddings.append(chunk["embedding"])
            added_count += 1

        logger.info(
            f"索引更新: 新增 {added_count} 个文档块, "
            f"总计 {len(self._documents)} 个"
        )

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        根据查询检索最相关的文档块。

        检索流程：
        1. 使用 embedding_engine 将查询转换为向量
        2. 计算查询向量与所有文档向量的余弦相似度
        3. 按相似度降序排列，返回 top_k 个结果

        Args:
            query: 查询文本
            top_k: 返回的最相关文档数量（默认 3）

        Returns:
            列表，每个元素为字典:
            - 'content' (str): 文档块内容
            - 'source' (str): 来源标识
            - 'score' (float): 余弦相似度分数
            - 'index' (int): 文档块序号

        Raises:
            RuntimeError: 未设置 embedding_engine 且索引为空
        """
        if not self._documents:
            logger.warning("检索索引为空，请先添加文档")
            return []

        if self.embedding_engine is None:
            raise RuntimeError(
                "未设置 embedding_engine，无法对查询进行向量化。"
                "请在初始化时传入 embedding_engine 参数。"
            )

        # 将查询转换为向量
        query_embedding = self.embedding_engine.embed_query(query)
        if not query_embedding:
            logger.error("查询向量化失败，返回空结果")
            return []

        # 计算与所有文档的相似度
        scores: list[tuple[int, float]] = []
        for i, doc_embedding in enumerate(self._embeddings):
            try:
                similarity = self.embedding_engine.cosine_similarity(
                    query_embedding, doc_embedding
                )
                scores.append((i, similarity))
            except Exception as e:
                logger.warning(
                    f"计算相似度失败 (文档 {i}): {type(e).__name__}: {e}"
                )
                continue

        # 按相似度降序排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 取 top_k 个结果
        results: list = []
        for idx, score in scores[:top_k]:
            doc = self._documents[idx]
            results.append({
                "content": doc["content"],
                "source": doc["source"],
                "score": round(score, 4),
                "index": doc["index"],
            })

        logger.info(
            f"检索完成: query='{query[:50]}...', "
            f"返回 {len(results)} 个结果, "
            f"最高分={results[0]['score'] if results else 'N/A'}"
        )
        return results

    def get_document_count(self) -> int:
        """返回索引中的文档块总数。"""
        return len(self._documents)

    def get_all_documents(self) -> list:
        """返回索引中所有文档块的副本。"""
        return list(self._documents)

    def clear(self) -> None:
        """清空索引中的所有文档块和向量。"""
        count = len(self._documents)
        self._documents.clear()
        self._embeddings.clear()
        logger.info(f"索引已清空，共移除 {count} 个文档块")
