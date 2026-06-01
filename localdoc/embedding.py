"""
LocalDoc Agent - 文本向量化引擎模块

负责将文本块和查询转换为向量表示，用于语义检索。

后端架构：
- 如果提供了 backend（如 NPU/GPU 后端），使用其 embed_texts 方法
- 如果没有提供 backend，使用 TF-IDF 作为轻量级回退方案
- TF-IDF 回退方案不需要任何外部模型或 GPU，适合开发和测试

余弦相似度计算作为静态方法提供，供检索器等其他模块使用。
"""

import math
from typing import Optional

from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    """
    文本向量化引擎 - 将文本转换为数值向量用于语义检索。

    支持两种工作模式：
    1. 后端模式：使用外部推理后端（NPU/GPU/CPU）生成高质量嵌入
    2. TF-IDF 回退模式：使用 TF-IDF 算法在本地计算轻量级向量

    Args:
        backend: 推理后端实例（可选），需实现 embed_texts(texts) -> list[list[float]] 方法

    用法示例：
        engine = EmbeddingEngine(backend=npu_backend)
        vectors = engine.embed_chunks(chunks)
        query_vec = engine.embed_query("什么是异构计算?")
    """

    def __init__(self, backend=None) -> None:
        self.backend = backend
        # TF-IDF 模式下的词汇表和 IDF 值
        self._vocab: dict[str, int] = {}  # 词 -> 索引
        self._idf: dict[str, float] = {}  # 词 -> IDF 值
        self._fitted = False  # 是否已经拟合过语料

        if backend is not None:
            logger.info(
                f"EmbeddingEngine 使用后端模式: {type(backend).__name__}"
            )
        else:
            logger.info(
                "EmbeddingEngine 使用 TF-IDF 回退模式（无需 GPU/外部模型）"
            )

    def embed_chunks(self, chunks: list) -> list:
        """
        对文本块列表进行向量化。

        Args:
            chunks: 文本块列表，每个元素为字典，需包含 'content' 字段

        Returns:
            向量列表，与输入 chunks 一一对应
        """
        if not chunks:
            logger.warning("输入的 chunks 列表为空")
            return []

        texts = [chunk["content"] for chunk in chunks]

        if self.backend is not None:
            vectors = self._embed_with_backend(texts)
        else:
            vectors = self._embed_with_tfidf(texts)

        logger.info(
            f"向量化完成: {len(chunks)} 个块, "
            f"维度={len(vectors[0]) if vectors else 0}"
        )
        return vectors

    def embed_query(self, query: str) -> list:
        """
        对查询文本进行向量化。

        生成的向量维度与 embed_chunks 保持一致，以便进行相似度计算。

        Args:
            query: 查询文本

        Returns:
            查询的向量表示（一维列表）
        """
        if not query or not query.strip():
            logger.warning("查询文本为空")
            return []

        if self.backend is not None:
            vectors = self._embed_with_backend([query])
            return vectors[0] if vectors else []
        else:
            vectors = self._embed_with_tfidf([query])
            return vectors[0] if vectors else []

    def _embed_with_backend(self, texts: list[str]) -> list:
        """
        使用后端进行向量化。

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        try:
            vectors = self.backend.embed_texts(texts)
            return vectors
        except Exception as e:
            logger.error(
                f"后端向量化失败，回退到 TF-IDF: {type(e).__name__}: {e}"
            )
            return self._embed_with_tfidf(texts)

    def _embed_with_tfidf(self, texts: list[str]) -> list:
        """
        使用 TF-IDF 算法进行向量化（回退方案）。

        TF-IDF (Term Frequency-Inverse Document Frequency) 是一种经典的
        文本表示方法。这里使用简化的词袋+IDF 权重方案。

        向量空间维度等于词汇表大小，每个维度对应一个词的 TF-IDF 权重。

        Args:
            texts: 文本列表

        Returns:
            TF-IDF 向量列表
        """
        # 如果未拟合或语料发生变化，重新构建词汇表和 IDF
        if not self._fitted:
            self._fit_tfidf(texts)

        vectors: list = []
        for text in texts:
            vector = self._compute_tfidf_vector(text)
            vectors.append(vector)

        return vectors

    def _fit_tfidf(self, corpus: list[str]) -> None:
        """
        基于语料库构建 TF-IDF 词汇表和 IDF 值。

        Args:
            corpus: 语料库文本列表
        """
        # 统计每个词出现在多少个文档中
        doc_freq: dict[str, int] = {}
        total_docs = len(corpus)

        for text in corpus:
            words = self._tokenize(text)
            # 每个文档中每个词只计算一次（用于计算 DF）
            unique_words = set(words)
            for word in unique_words:
                doc_freq[word] = doc_freq.get(word, 0) + 1

        # 构建词汇表：按字母排序以保证确定性
        sorted_words = sorted(doc_freq.keys())
        self._vocab = {word: idx for idx, word in enumerate(sorted_words)}

        # 计算 IDF 值: log(N / df)，其中 N 是文档总数，df 是包含该词的文档数
        self._idf = {}
        for word, df in doc_freq.items():
            # 使用平滑 IDF 避免除零: log((N+1) / (df+1)) + 1
            self._idf[word] = math.log((total_docs + 1) / (df + 1)) + 1

        self._fitted = True
        logger.debug(
            f"TF-IDF 词汇表构建完成: 词汇量={len(self._vocab)}, "
            f"文档数={total_docs}"
        )

    def _compute_tfidf_vector(self, text: str) -> list:
        """
        计算单个文本的 TF-IDF 向量。

        Args:
            text: 输入文本

        Returns:
            TF-IDF 向量（长度等于词汇表大小）
        """
        vector = [0.0] * len(self._vocab)
        words = self._tokenize(text)

        if not words:
            return vector

        # 统计词频（TF）
        word_count: dict[str, int] = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

        # 计算 TF-IDF 权重
        for word, count in word_count.items():
            if word in self._vocab:
                tf = count / len(words)  # 词频归一化
                idf = self._idf.get(word, 1.0)
                idx = self._vocab[word]
                vector[idx] = tf * idf

        # L2 归一化，使向量长度为 1（方便余弦相似度计算）
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    @staticmethod
    def _tokenize(text: str) -> list:
        """
        简单的中英文分词器。

        英文按空格和标点分词并转小写；
        中文按字符级别分词（每个汉字作为一个 token）。

        Args:
            text: 输入文本

        Returns:
            token 列表
        """
        tokens: list[str] = []
        # 使用正则按空白和标点分割
        import re
        # 将中文字符和英文单词分开处理
        segments = re.findall(r'[一-鿿]|[a-zA-Z0-9]+', text)
        for seg in segments:
            if '一' <= seg[0] <= '鿿':
                # 中文字符：逐字作为 token
                tokens.append(seg)
            else:
                # 英文单词：转小写
                tokens.append(seg.lower())
        return tokens

    @staticmethod
    def cosine_similarity(vec_a: list, vec_b: list) -> float:
        """
        计算两个向量的余弦相似度。

        余弦相似度衡量两个向量方向的接近程度：
        - 1.0 表示完全相同
        - 0.0 表示正交（无关）
        - -1.0 表示完全相反

        Args:
            vec_a: 向量 A
            vec_b: 向量 B

        Returns:
            余弦相似度值（-1.0 到 1.0）

        Raises:
            ValueError: 向量维度不匹配或向量为空
        """
        if len(vec_a) != len(vec_b):
            raise ValueError(
                f"向量维度不匹配: {len(vec_a)} vs {len(vec_b)}"
            )
        if len(vec_a) == 0:
            raise ValueError("向量为空，无法计算相似度")

        # 点积
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        # 向量模长
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        # 避免除零
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
