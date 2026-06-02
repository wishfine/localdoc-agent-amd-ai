"""
CPU 后端实现 - LocalDoc Agent 基线计算引擎

嵌入采用 TF-IDF（纯 Python，无第三方依赖）。

TF-IDF 生命周期：
- fit_and_embed(texts): 在 texts 上构建词汇表和 IDF，返回所有 texts 的向量
- transform(texts): 用已有词汇表和 IDF 计算向量（不修改语料库，用于查询）
- reset_corpus(): 清空语料库（摄入新批次文档前调用）
"""

import math
import time
import re
from collections import Counter
from typing import List, Optional

from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class CPUBackend:
    """CPU 计算后端，TF-IDF 嵌入 + 关键词匹配问答。"""

    def __init__(self) -> None:
        self._corpus: List[str] = []
        self._vocab: Optional[List[str]] = None
        self._vocab_index: Optional[dict] = None
        self._idf: Optional[List[float]] = None

    @property
    def name(self) -> str:
        return "CPU"

    def is_available(self) -> bool:
        return True

    def reset_corpus(self) -> None:
        """清空语料库和词汇表。"""
        self._corpus = []
        self._vocab = None
        self._vocab_index = None
        self._idf = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词器：中文逐字拆分，英文按空格/标点拆分。"""
        chinese_chars = re.findall(r'[一-鿿]', text)
        latin_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return chinese_chars + latin_tokens

    def fit_and_embed(self, texts: List[str]) -> List[List[float]]:
        """
        在 texts 上构建 TF-IDF 词汇表，返回所有 texts 的向量。

        调用后词汇表和 IDF 冻结。后续 transform() 使用同一词汇表。
        """
        if not texts:
            return []

        self._corpus = list(texts)
        tokenized = [self._tokenize(t) for t in self._corpus]

        # 构建词汇表
        vocab_set: set = set()
        for tokens in tokenized:
            vocab_set.update(tokens)
        self._vocab = sorted(vocab_set)
        self._vocab_index = {w: i for i, w in enumerate(self._vocab)}
        vocab_size = len(self._vocab)
        n_docs = len(self._corpus)

        # 计算 IDF
        doc_freq: Counter = Counter()
        for tokens in tokenized:
            for t in set(tokens):
                doc_freq[t] += 1
        self._idf = [0.0] * vocab_size
        for word, idx in self._vocab_index.items():
            self._idf[idx] = math.log((n_docs + 1) / (1 + doc_freq[word])) + 1.0

        # 计算向量
        vectors = self._compute_vectors(tokenized)

        logger.info("CPU 后端：fit %d 条语料，词汇表 %d 维", n_docs, vocab_size)
        return vectors

    def transform(self, texts: List[str]) -> List[List[float]]:
        """
        用已有词汇表和 IDF 计算 texts 的向量。

        不修改语料库。用于查询阶段。
        如果词汇表未构建，自动调用 fit_and_embed。
        """
        if not texts:
            return []

        if self._vocab is None:
            return self.fit_and_embed(texts)

        tokenized = [self._tokenize(t) for t in texts]
        return self._compute_vectors(tokenized)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        兼容接口：如果词汇表已存在则 transform，否则 fit_and_embed。
        """
        if self._vocab is not None:
            return self.transform(texts)
        return self.fit_and_embed(texts)

    def _compute_vectors(self, tokenized: List[List[str]]) -> List[List[float]]:
        """使用当前词汇表和 IDF 计算向量。"""
        vocab_size = len(self._vocab)
        vectors: List[List[float]] = []
        for tokens in tokenized:
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

    def get_corpus_texts(self) -> List[str]:
        return list(self._corpus)

    # ---------- 问答生成 ----------

    def generate_answer(self, query: str, context, max_length: int = 512) -> str:
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

    # ---------- 基准测试 ----------

    def benchmark_embedding(self, texts: List[str]) -> dict:
        start = time.perf_counter()
        vectors = self.fit_and_embed(texts)
        elapsed = time.perf_counter() - start
        dim = len(vectors[0]) if vectors else 0
        return {
            "backend": self.name,
            "num_texts": len(texts),
            "elapsed_seconds": round(elapsed, 6),
            "vectors_per_second": round(len(texts) / elapsed, 2) if elapsed > 0 else 0,
            "embedding_dim": dim,
        }

    def benchmark_generation(self, query: str, context: List[str]) -> dict:
        start = time.perf_counter()
        answer = self.generate_answer(query, context)
        elapsed = time.perf_counter() - start
        return {
            "backend": self.name,
            "elapsed_seconds": round(elapsed, 6),
            "answer_length": len(answer),
            "num_context_docs": len(context),
        }
