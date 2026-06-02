"""
模拟 NPU 后端 - 仅用于演示和教学目的

================================================================================
** 重要免责声明 **
================================================================================
此模块提供的是 **模拟的** NPU 行为，而非真实的 NPU 硬件加速。

- 所有计算仍然在 CPU 上执行
- 通过 time.sleep() 添加人为延迟来模拟 NPU 的推理耗时
- 输出结果与真实 NPU 无关，仅用于演示异构计算框架的接口设计

** 不得将此后端的性能数据或结果作为真实 NPU 性能的参考！ **
** 如需真实 NPU 性能数据，请使用 AMDNPUBackend 并配合真实硬件。 **
================================================================================

使用场景：
    1. 在没有 NPU 硬件的设备上演示完整的异构计算流程
    2. 开发和调试上层应用逻辑，无需等待真实硬件就绪
    3. 课堂演示 CPU / GPU / NPU 三后端对比的框架设计

典型用法：
    backend = SimulatedNPUBackend()
    # 后端始终可用，但每次调用都会打印警告
    vectors = backend.embed_texts(["演示文本"])
    answer = backend.generate_answer("问题", ["上下文"])
"""

import time
import math
import re
import random
from collections import Counter
from typing import List, Dict, Any

from localdoc.utils.logger import get_logger

logger = get_logger(__name__)

# 模拟延迟参数（秒）
# 这些值是为了在演示中产生可观测的时间差，
# 并不代表真实 NPU 的性能特征
_SIMULATED_EMBED_DELAY_PER_TEXT = 0.005   # 每条文本 5ms 模拟延迟
_SIMULATED_GENERATE_DELAY = 0.02           # 问答生成 20ms 模拟延迟
_SIMULATED_JITTER_MAX = 0.003              # 随机抖动上限 3ms

# 警告信息 —— 每次实例化时打印
_SIMULATION_WARNING = (
    "\n"
    "=" * 72 + "\n"
    "  [警告] 当前使用的是 SimulatedNPUBackend（模拟 NPU 后端）\n"
    "  所有计算仍在 CPU 上执行，人为延迟仅用于模拟 NPU 推理耗时。\n"
    "  此后端的性能数据不代表真实 AMD NPU 的性能。\n"
    "  如需真实 NPU 性能，请使用 AMDNPUBackend + 真实硬件。\n"
    "=" * 72
)


class SimulatedNPUBackend:
    """
    模拟 NPU 后端 —— 仅用于演示

    ** 此后端不是真正的 NPU 加速 **

    所有计算在 CPU 上完成，然后通过 sleep() 添加人为延迟
    来模拟 NPU 推理的时间特性。目的仅为演示异构计算框架中
    多后端切换的接口设计，以及提供一个可供对比的模拟时间基线。

    警告：
        不得将此后端的 benchmark 结果作为真实 NPU 性能数据发表或引用。
    """

    def __init__(self) -> None:
        """初始化模拟 NPU 后端，打印醒目警告。"""
        print(_SIMULATION_WARNING)
        logger.warning(
            "SimulatedNPUBackend 已初始化。"
            "这是一个模拟后端，所有计算在 CPU 上执行。"
            "请勿将结果作为真实 NPU 性能数据。"
        )

    # ---------- 属性 ----------

    @property
    def name(self) -> str:
        """返回后端名称标识，明确标注为模拟。"""
        return "Simulated NPU (Demo Only)"

    # ---------- 可用性检查 ----------

    def is_available(self) -> bool:
        """
        检查模拟 NPU 后端是否可用。

        模拟后端始终可用，因为它不需要任何特殊硬件。

        Returns:
            始终返回 True
        """
        return True

    # ---------- 文本嵌入 ----------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        模拟 NPU 文本嵌入。

        实际使用 CPU 进行 TF-IDF 计算，然后添加人为延迟模拟 NPU 推理。
        延迟 = 基础延迟 * 文本数量 + 随机抖动。

        !!! 警告：此方法不使用真正的 NPU !!!
        !!! 输出结果仅为 CPU TF-IDF 计算结果 !!!

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 每条文本对应的 TF-IDF 向量（CPU 计算结果）
        """
        if not texts:
            return []

        # 打印警告：模拟模式
        logger.warning(
            "[模拟 NPU] embed_texts 被调用，实际在 CPU 上执行。"
            "文本数 = %d",
            len(texts),
        )

        # ---------- 实际 CPU 计算 ----------
        tokenized_docs = [self._tokenize(t) for t in texts]
        vocab_set: set = set()
        for tokens in tokenized_docs:
            vocab_set.update(tokens)
        feature_names: List[str] = sorted(vocab_set)
        vocab_index = {w: i for i, w in enumerate(feature_names)}
        vocab_size = len(feature_names)
        n_docs = len(texts)

        doc_freq: Counter = Counter()
        for tokens in tokenized_docs:
            for t in set(tokens):
                doc_freq[t] += 1
        idf_values = [0.0] * vocab_size
        for word, idx in vocab_index.items():
            idf_values[idx] = math.log((n_docs + 1) / (1 + doc_freq[word])) + 1.0

        vectors: List[List[float]] = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec = [0.0] * vocab_size
            for word, count in tf.items():
                if word in vocab_index:
                    i = vocab_index[word]
                    vec[i] = (count / total) * idf_values[i]
            # L2 归一化
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)

        # ---------- 添加模拟延迟 ----------
        delay = _SIMULATED_EMBED_DELAY_PER_TEXT * len(texts)
        jitter = random.uniform(0, _SIMULATED_JITTER_MAX)
        total_delay = delay + jitter
        logger.info(
            "[模拟 NPU] 添加 %.4f 秒模拟延迟（基础 %.4f + 抖动 %.4f）",
            total_delay, delay, jitter,
        )
        time.sleep(total_delay)

        logger.info("[模拟 NPU] 嵌入完成（CPU 实际计算 + 模拟延迟）")
        return vectors

    # ---------- 问答生成 ----------

    def generate_answer(
        self,
        query: str,
        context,
        max_length: int = 512,
    ) -> str:
        """
        模拟 NPU 问答生成。

        实际使用 CPU 进行关键词匹配抽取式问答，
        然后添加人为延迟模拟 NPU 推理。

        Args:
            query: 用户提出的问题
            context: 上下文，可以是字符串或字符串列表
            max_length: 答案最大字符数

        Returns:
            str: 生成的答案文本（CPU 关键词匹配结果）
        """
        # 统一处理：将 context 转为字符串列表
        if isinstance(context, str):
            context_list = [context]
        elif isinstance(context, list):
            context_list = context
        else:
            context_list = [str(context)]

        if not context_list or all(not c for c in context_list):
            logger.warning("[模拟 NPU] 未提供上下文，无法生成答案")
            return "抱歉，未找到相关上下文信息，无法回答该问题。"

        logger.warning(
            "[模拟 NPU] generate_answer 被调用，实际在 CPU 上执行。"
        )

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return "抱歉，无法解析您的问题，请尝试重新表述。"

        # ---------- CPU 关键词匹配 ----------
        all_sentences: List[str] = []
        for doc in context_list:
            sentences = re.split(r'[。！？.!?\n]+', doc)
            for s in sentences:
                s = s.strip()
                if s:
                    all_sentences.append(s)

        if not all_sentences:
            return context_list[0][:max_length]

        best_sentence = ""
        best_score = 0
        for sentence in all_sentences:
            sentence_tokens = set(self._tokenize(sentence))
            score = len(query_tokens & sentence_tokens)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        answer = best_sentence[:max_length] if best_sentence else context[0][:max_length]

        # ---------- 添加模拟延迟 ----------
        jitter = random.uniform(0, _SIMULATED_JITTER_MAX)
        total_delay = _SIMULATED_GENERATE_DELAY + jitter
        logger.info(
            "[模拟 NPU] 添加 %.4f 秒模拟延迟",
            total_delay,
        )
        time.sleep(total_delay)

        logger.info("[模拟 NPU] 问答完成（CPU 实际计算 + 模拟延迟）")
        return answer

    # ---------- 设备信息 ----------

    def get_device_info(self) -> Dict[str, Any]:
        """
        获取模拟 NPU 后端信息。

        Returns:
            dict: 标明这是模拟后端，非真实 NPU 硬件
        """
        return {
            "backend": self.name,
            "available": True,
            "is_simulation": True,
            "warning": "这是模拟后端，所有计算在 CPU 上执行，不代表真实 NPU 性能",
            "simulated_embed_delay_per_text": _SIMULATED_EMBED_DELAY_PER_TEXT,
            "simulated_generate_delay": _SIMULATED_GENERATE_DELAY,
        }

    # ---------- 辅助方法 ----------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        简单分词器：中文逐字拆分，英文按空格/标点拆分。

        Args:
            text: 输入文本

        Returns:
            分词后的 token 列表
        """
        chinese_chars = re.findall(r'[一-鿿]', text)
        latin_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return chinese_chars + latin_tokens

    # ---------- 性能基准测试 ----------

    def benchmark_embedding(self, texts: List[str]) -> dict:
        """
        对 embed_texts 进行计时基准测试。

        注意：结果中包含模拟延迟，不代表真实 NPU 性能。

        Args:
            texts: 测试用文本列表

        Returns:
            dict: 包含 backend、num_texts、elapsed_seconds、
                  vectors_per_second、embedding_dim、is_simulation 等字段
        """
        logger.info("[模拟 NPU] 开始嵌入基准测试，文本数 = %d", len(texts))

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
            "is_simulation": True,
            "warning": "此数据包含模拟延迟，不代表真实 NPU 性能",
        }
        logger.info("[模拟 NPU] 嵌入基准测试完成 - %.4f 秒", elapsed)
        return result

    def benchmark_generation(
        self,
        query: str,
        context: List[str],
    ) -> dict:
        """
        对 generate_answer 进行计时基准测试。

        注意：结果中包含模拟延迟，不代表真实 NPU 性能。

        Args:
            query: 测试查询
            context: 上下文文档列表

        Returns:
            dict: 包含 backend、elapsed_seconds、answer_length、
                  num_context_docs、is_simulation 等字段
        """
        logger.info("[模拟 NPU] 开始生成基准测试")

        start = time.perf_counter()
        answer = self.generate_answer(query, context)
        elapsed = time.perf_counter() - start

        result = {
            "backend": self.name,
            "elapsed_seconds": round(elapsed, 6),
            "answer_length": len(answer),
            "num_context_docs": len(context),
            "is_simulation": True,
            "warning": "此数据包含模拟延迟，不代表真实 NPU 性能",
        }
        logger.info("[模拟 NPU] 生成基准测试完成 - %.4f 秒", elapsed)
        return result
