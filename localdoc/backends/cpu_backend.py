"""
CPU 后端实现 - LocalDoc Agent 基线计算引擎

本模块提供基于纯 CPU 的文本嵌入和问答生成能力，
作为异构计算实验中的对照基线。

嵌入方法采用 TF-IDF 思路的纯 Python 实现（collections.Counter），
无需安装 sklearn 等第三方依赖即可运行。
若环境中已安装 sklearn，则可选择使用更高效的实现。

典型用途：
    backend = CPUBackend()
    if backend.is_available():
        vectors = backend.embed_texts(["你好世界", "异构计算"])
        answer = backend.generate_answer("什么是异构计算?", context_docs)
"""

import math
import time
import re
from collections import Counter
from typing import List, Optional

from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class CPUBackend:
    """
    CPU 计算后端

    作为异构计算实验的基线对照组，提供：
    - 基于 TF-IDF 的文本嵌入（纯 Python 实现）
    - 基于关键词匹配的抽取式问答

    此后端始终可用，不依赖任何特殊硬件或第三方库。
    """

    # ---------- 属性 ----------

    @property
    def name(self) -> str:
        """返回后端名称标识。"""
        return "CPU"

    # ---------- 可用性检查 ----------

    def is_available(self) -> bool:
        """
        检查 CPU 后端是否可用。

        CPU 后端始终可用，因为每台设备都有 CPU。

        Returns:
            始终返回 True
        """
        return True

    # ---------- 文本嵌入 ----------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        简单分词器：按非字母数字字符切分，转小写。

        对中文文本，按单字切分；对英文/数字按空格和标点切分。

        Args:
            text: 输入文本

        Returns:
            分词后的 token 列表
        """
        # 中文字符：逐字拆分（中文无天然空格分隔）
        chinese_chars = re.findall(r'[一-鿿]', text)
        # 英文和数字 token
        latin_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return chinese_chars + latin_tokens

    def _build_tfidf(self, texts: List[str]) -> tuple:
        """
        为一组文本构建 TF-IDF 向量。

        Args:
            texts: 文本列表

        Returns:
            (vectors, feature_names):
                vectors - List[List[float]]，每条文本的 TF-IDF 向量
                feature_names - 全局词汇表（有序）
        """
        # 对每条文本分词并统计词频
        tokenized_docs: List[List[str]] = [self._tokenize(t) for t in texts]

        # 构建全局词汇表
        vocab_set: set = set()
        for tokens in tokenized_docs:
            vocab_set.update(tokens)
        feature_names: List[str] = sorted(vocab_set)
        vocab_index = {word: i for i, word in enumerate(feature_names)}
        vocab_size = len(feature_names)
        n_docs = len(texts)

        # 计算 IDF：log(N / (1 + df))
        doc_freq: Counter = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freq[t] += 1

        idf = [0.0] * vocab_size
        for word, idx in vocab_index.items():
            idf[idx] = math.log((n_docs + 1) / (1 + doc_freq[word])) + 1.0

        # 计算每条文档的 TF-IDF 向量
        vectors: List[List[float]] = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec = [0.0] * vocab_size
            for word, count in tf.items():
                if word in vocab_index:
                    idx = vocab_index[word]
                    vec[idx] = (count / total) * idf[idx]

            # L2 归一化
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]

            vectors.append(vec)

        return vectors, feature_names

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        使用纯 Python TF-IDF 方法将文本转换为向量表示。

        此方法不依赖任何第三方库，可在任意环境下运行。
        作为异构计算实验的 CPU 基线。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 每条文本对应的 TF-IDF 向量，
            所有向量维度一致（等于全局词汇表大小）

        Example:
            >>> backend = CPUBackend()
            >>> vecs = backend.embed_texts(["hello world", "world peace"])
            >>> len(vecs) == 2
            True
        """
        if not texts:
            return []

        logger.info("CPU 后端：开始对 %d 条文本进行 TF-IDF 嵌入", len(texts))
        vectors, features = self._build_tfidf(texts)
        logger.info(
            "CPU 后端：嵌入完成，向量维度 = %d", len(features)
        )
        return vectors

    # ---------- 问答生成 ----------

    def generate_answer(
        self,
        query: str,
        context: List[str],
        max_length: int = 512,
    ) -> str:
        """
        基于关键词匹配的抽取式问答。

        从 context 中选择与 query 关键词重叠最多的句子作为答案。
        这是一个简单的基线实现，用于与 GPU/NPU 后端对比。

        Args:
            query: 用户提出的问题
            context: 上下文文档列表（每条为一段文本）
            max_length: 答案最大字符数

        Returns:
            str: 生成的答案文本；若无匹配上下文则返回默认提示
        """
        if not context:
            logger.warning("CPU 后端：未提供上下文，无法生成答案")
            return "抱歉，未找到相关上下文信息，无法回答该问题。"

        logger.info("CPU 后端：开始基于关键词匹配生成答案")

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return "抱歉，无法解析您的问题，请尝试重新表述。"

        # 将每条上下文切分为句子，然后对每个句子打分
        best_sentence = ""
        best_score = 0

        for doc in context:
            # 按中英文标点和换行切分句子
            sentences = re.split(r'[。！？.!?\n]+', doc)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_tokens = set(self._tokenize(sentence))
                # 评分 = query 关键词在句子中的命中数
                score = len(query_tokens & sentence_tokens)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        if best_sentence:
            # 截断到最大长度
            answer = best_sentence[:max_length]
            logger.info("CPU 后端：找到匹配句子，关键词命中数 = %d", best_score)
            return answer

        logger.info("CPU 后端：未找到高度匹配的句子，返回上下文首句")
        # 退化策略：返回第一条上下文的前 max_length 个字符
        fallback = context[0][:max_length]
        return fallback

    # ---------- 性能基准测试 ----------

    def benchmark_embedding(self, texts: List[str]) -> dict:
        """
        对 embed_texts 进行计时基准测试。

        Args:
            texts: 测试用文本列表

        Returns:
            dict: 包含以下字段：
                - backend: 后端名称
                - num_texts: 文本数量
                - elapsed_seconds: 耗时（秒）
                - vectors_per_second: 吞吐量
                - embedding_dim: 输出向量维度
        """
        logger.info("CPU 后端：开始嵌入基准测试，文本数 = %d", len(texts))

        start = time.perf_counter()
        vectors = self.embed_texts(texts)
        elapsed = time.perf_counter() - start

        dim = len(vectors[0]) if vectors else 0
        throughput = len(texts) / elapsed if elapsed > 0 else float("inf")

        result = {
            "backend": self.name,
            "num_texts": len(texts),
            "elapsed_seconds": round(elapsed, 6),
            "vectors_per_second": round(throughput, 2),
            "embedding_dim": dim,
        }
        logger.info("CPU 后端：嵌入基准测试完成 - %.4f 秒", elapsed)
        return result

    def benchmark_generation(
        self,
        query: str,
        context: List[str],
    ) -> dict:
        """
        对 generate_answer 进行计时基准测试。

        Args:
            query: 测试查询
            context: 上下文文档列表

        Returns:
            dict: 包含以下字段：
                - backend: 后端名称
                - elapsed_seconds: 耗时（秒）
                - answer_length: 答案长度（字符数）
                - num_context_docs: 上下文文档数量
        """
        logger.info("CPU 后端：开始生成基准测试")

        start = time.perf_counter()
        answer = self.generate_answer(query, context)
        elapsed = time.perf_counter() - start

        result = {
            "backend": self.name,
            "elapsed_seconds": round(elapsed, 6),
            "answer_length": len(answer),
            "num_context_docs": len(context),
        }
        logger.info("CPU 后端：生成基准测试完成 - %.4f 秒", elapsed)
        return result
