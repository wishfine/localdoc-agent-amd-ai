"""
CPU 后端实现 - LocalDoc Agent 基线计算引擎

本模块提供基于纯 CPU 的文本嵌入和问答生成能力，
作为异构计算实验中的对照基线。

嵌入方法采用 TF-IDF 思路的纯 Python 实现（collections.Counter），
无需安装 sklearn 等第三方依赖即可运行。

TF-IDF 生命周期：
- fit_corpus(corpus): 在语料库上构建词汇表和 IDF（摄入阶段调用一次）
- embed_texts(texts): 使用已冻结的词汇表将文本转为向量（查询阶段调用）
- 如果未调用 fit_corpus，embed_texts 会自动调用
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

    TF-IDF 生命周期：
    1. fit_corpus(corpus): 在摄入的全部文档上构建词汇表和 IDF 值
    2. embed_texts(texts): 使用冻结的词汇表计算向量（查询时不扩展词表）
    """

    def __init__(self) -> None:
        self._corpus: List[str] = []
        self._vocab: Optional[List[str]] = None
        self._idf: Optional[List[float]] = None
        self._vocab_index: Optional[dict] = None
        self._vocab_frozen: bool = False

    @property
    def name(self) -> str:
        return "CPU"

    def is_available(self) -> bool:
        return True

    # ---------- 文本嵌入 ----------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词器：中文逐字拆分，英文按空格/标点拆分。"""
        chinese_chars = re.findall(r'[一-鿿]', text)
        latin_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return chinese_chars + latin_tokens

    def fit_corpus(self, corpus: List[str]) -> None:
        """
        在语料库上构建 TF-IDF 词汇表和 IDF 值。

        调用后词汇表冻结，后续 embed_texts 不会扩展词表。
        这保证了摄入阶段和查询阶段的向量维度一致。

        Args:
            corpus: 完整的文档文本列表
        """
        self._corpus = list(corpus)
        self._vocab_frozen = False

        tokenized = [self._tokenize(t) for t in self._corpus]

        vocab_set: set = set()
        for tokens in tokenized:
            vocab_set.update(tokens)
        feature_names: List[str] = sorted(vocab_set)
        self._vocab_index = {word: i for i, word in enumerate(feature_names)}
        self._vocab = feature_names
        vocab_size = len(feature_names)

        n_docs = len(self._corpus)
        doc_freq: Counter = Counter()
        for tokens in tokenized:
            for t in set(tokens):
                doc_freq[t] += 1

        self._idf = [0.0] * vocab_size
        for word, idx in self._vocab_index.items():
            self._idf[idx] = math.log((n_docs + 1) / (1 + doc_freq[word])) + 1.0

        self._vocab_frozen = True
        logger.info("CPU 后端：词汇表构建完成，维度 = %d，文档数 = %d", vocab_size, n_docs)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        使用 TF-IDF 将文本转为向量。

        如果词汇表已冻结（fit_corpus 已调用），使用冻结词表计算向量，
        不扩展词表。这保证查询向量与文档向量维度一致。

        如果词汇表未冻结（fit_corpus 未调用），自动构建词表。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            向量列表，所有向量维度一致
        """
        if not texts:
            return []

        if not self._vocab_frozen:
            self.fit_corpus(texts)

        logger.info("CPU 后端：嵌入 %d 条文本，向量维度 = %d", len(texts), len(self._vocab))
        return self._compute_vectors(texts)

    def _compute_vectors(self, texts: List[str]) -> List[List[float]]:
        """使用当前冻结的词汇表计算向量。"""
        vocab_size = len(self._vocab)
        vectors: List[List[float]] = []

        for text in texts:
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec = [0.0] * vocab_size
            for word, count in tf.items():
                if word in self._vocab_index:
                    idx = self._vocab_index[word]
                    vec[idx] = (count / total) * self._idf[idx]
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)

        return vectors

    # ---------- 问答生成 ----------

    def generate_answer(
        self,
        query: str,
        context,
        max_length: int = 512,
    ) -> str:
        """
        基于关键词匹配的抽取式问答。

        Args:
            query: 用户提出的问题
            context: 上下文，可以是字符串或字符串列表
            max_length: 答案最大字符数
        """
        if isinstance(context, str):
            context_list = [context]
        elif isinstance(context, list):
            context_list = context
        else:
            context_list = [str(context)]

        if not context_list or all(not c for c in context_list):
            return "抱歉，未找到相关上下文信息，无法回答该问题。"

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return "抱歉，无法解析您的问题，请尝试重新表述。"

        best_sentence = ""
        best_score = 0

        for doc in context_list:
            sentences = re.split(r'[。！？.!?\n]+', doc)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_tokens = set(self._tokenize(sentence))
                score = len(query_tokens & sentence_tokens)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        if best_sentence:
            return best_sentence[:max_length]

        return context_list[0][:max_length]

    # ---------- 性能基准测试 ----------

    def benchmark_embedding(self, texts: List[str]) -> dict:
        """对 embed_texts 进行计时基准测试。"""
        start = time.perf_counter()
        vectors = self.embed_texts(texts)
        elapsed = time.perf_counter() - start

        dim = len(vectors[0]) if vectors else 0
        throughput = len(texts) / elapsed if elapsed > 0 else float("inf")

        return {
            "backend": self.name,
            "num_texts": len(texts),
            "elapsed_seconds": round(elapsed, 6),
            "vectors_per_second": round(throughput, 2),
            "embedding_dim": dim,
        }

    def benchmark_generation(self, query: str, context: List[str]) -> dict:
        """对 generate_answer 进行计时基准测试。"""
        start = time.perf_counter()
        answer = self.generate_answer(query, context)
        elapsed = time.perf_counter() - start

        return {
            "backend": self.name,
            "elapsed_seconds": round(elapsed, 6),
            "answer_length": len(answer),
            "num_context_docs": len(context),
        }
