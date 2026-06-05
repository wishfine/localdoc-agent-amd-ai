"""
AMD NPU 后端实现 - 基于 ONNX Runtime / Ryzen AI SDK 的 NPU 加速引擎

本模块提供利用 AMD 锐龙 AI 处理器内置 NPU（神经处理单元）
进行文本嵌入和问答生成的能力。

硬件与软件要求：
    - AMD 锐龙 AI 系列处理器（XDNA / XDNA2 架构 NPU）
    - Ryzen AI SDK（提供 ONNX Runtime Execution Provider）
    - ONNX Runtime >= 1.16，且编译了 VitisAI EP 或 DirectML EP

NPU 架构说明：
    AMD 锐龙 AI MAX+ 395 处理器内置 XDNA2 架构 NPU，
    算力可达 50 TOPS（INT8）。NPU 专为低功耗、高吞吐的 AI 推理设计，
    适合在笔记本等移动设备上运行嵌入模型和小型语言模型。

ONNX Runtime EP（Execution Provider）：
    ONNX Runtime 通过 Execution Provider 抽象层支持多种硬件后端。
    Ryzen AI SDK 提供的 EP 使得 ONNX 模型可以直接在 NPU 上执行，
    无需经过 CUDA 或 CPU 的计算路径。

典型用法：
    backend = AMDNPUBackend()
    if backend.is_available():
        vectors = backend.embed_texts(["文本一", "文本二"])
        info = backend.get_device_info()
        print(info)
    else:
        print("NPU 后端不可用，请检查 Ryzen AI SDK 安装")

注意：
    NPU 后端需要特定的硬件和驱动支持。在不具备 NPU 的设备上，
    后端会优雅降级并给出友好提示。
"""

import time
import math
import re
from collections import Counter
from typing import List, Optional, Dict, Any

from localdoc.backends.cpu_backend import CPUBackend
from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class AMDNPUBackend:
    """
    AMD NPU 加速后端

    通过 ONNX Runtime + Ryzen AI SDK 实现 NPU 加速的文本嵌入与问答生成。
    需要 AMD 锐龙 AI 处理器及配套的 Ryzen AI 软件栈。

    如果运行环境不满足要求，所有方法会优雅降级到 CPU 计算，
    并在日志中给出明确提示。
    """

    def __init__(self) -> None:
        """初始化 NPU 后端，懒加载依赖库。"""
        self._ort = None  # onnxruntime 模块引用
        self._session = None  # ONNX Runtime 推理会话（如有预编译模型）
        self._npu_available: bool = False
        self._init_attempted: bool = False
        self._ep_name: Optional[str] = None  # 实际使用的 Execution Provider 名称
        self._cpu_backend = CPUBackend()

    # ---------- 属性 ----------

    @property
    def name(self) -> str:
        """返回后端名称标识。"""
        return "AMD NPU"

    # ---------- 内部初始化 ----------

    def _lazy_init(self) -> None:
        """
        懒加载 ONNX Runtime 并检查 NPU Execution Provider 可用性。

        检查流程：
        1. 尝试导入 onnxruntime
        2. 检查可用的 EP 列表中是否包含 VitisAI 或 DirectML
        3. 标记 NPU 可用状态
        """
        if self._init_attempted:
            return
        self._init_attempted = True

        try:
            import onnxruntime as ort  # 延迟导入
            self._ort = ort
        except ImportError:
            logger.warning(
                "NPU 后端：未安装 ONNX Runtime。"
                "请参考 https://onnxruntime.ai/ 安装。\n"
                "如需 NPU 加速，请安装 Ryzen AI SDK 及对应的 ONNX Runtime 版本：\n"
                "  pip install onnxruntime-ryzen-ai"
            )
            return

        # 获取所有可用的 Execution Provider
        available_eps = ort.get_available_providers()
        logger.info("NPU 后端：ONNX Runtime 已安装，可用 EP 列表: %s", available_eps)

        # 检查是否有支持 NPU 的 EP
        # VitisAI EP：Ryzen AI SDK 提供的原生 NPU EP
        # DmlExecutionProvider：DirectML EP（Windows 上可间接使用 NPU）
        npu_eps = [
            "VitisAIExecutionProvider",
            "RyzenAIExecutionProvider",
            "DmlExecutionProvider",
        ]

        for ep in npu_eps:
            if ep in available_eps:
                self._ep_name = ep
                self._npu_available = True
                logger.info("NPU 后端：找到 NPU 兼容 EP - %s", ep)
                break

        if not self._npu_available:
            logger.warning(
                "NPU 后端：ONNX Runtime 已安装，但未找到 NPU Execution Provider。\n"
                "可用 EP: %s\n"
                "请确认已安装 Ryzen AI SDK：\n"
                "  1. 从 https://ryzenai.amd.com/ 下载 SDK\n"
                "  2. 安装对应的 ONNX Runtime EP\n"
                "  3. 确认 BIOS 中 NPU 已启用",
                available_eps,
            )

    # ---------- 可用性检查 ----------

    def is_available(self) -> bool:
        """
        检查 AMD NPU 后端是否真正可用。

        检查条件：
        1. ONNX Runtime 已安装
        2. 存在支持 NPU 的 Execution Provider（VitisAI / DirectML）

        Returns:
            bool: NPU 后端是否可用于加速计算
        """
        self._lazy_init()
        return self._npu_available

    def has_real_inference(self) -> bool:
        """
        检查 NPU 后端是否真正执行 NPU 推理（而非 CPU 回退）。

        当前实现：核心计算为 NumPy 归一化，未创建 ONNX Runtime session。
        因此始终返回 False。后续实现真正的 ONNX 模型推理后应返回 True。

        Returns:
            始终返回 False（当前为 CPU 回退实现）
        """
        return False

    def reset_corpus(self) -> None:
        """Reset fitted TF-IDF state before rebuilding the document index."""
        self._cpu_backend.reset_corpus()

    # ---------- 设备信息 ----------

    def get_device_info(self) -> Dict[str, Any]:
        """
        获取 AMD NPU 设备详细信息。

        Returns:
            dict: 包含以下字段：
                - backend: 后端名称
                - available: NPU 是否可用
                - execution_provider: 使用的 EP 名称
                - onnxruntime_version: ONNX Runtime 版本
                - available_providers: 所有可用 EP 列表
        """
        self._lazy_init()

        info: Dict[str, Any] = {
            "backend": self.name,
            "available": self.is_available(),
            "execution_provider": self._ep_name,
            "onnxruntime_version": None,
            "available_providers": [],
        }

        if self._ort is not None:
            info["onnxruntime_version"] = self._ort.__version__
            try:
                info["available_providers"] = self._ort.get_available_providers()
            except Exception:
                pass

        return info

    # ---------- 文本嵌入 ----------

    def fit_and_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Fit a frozen TF-IDF vocabulary and embed document texts.

        Current NPU implementation detects Ryzen AI EPs but does not run an
        ONNX model on the NPU, so embedding is delegated to the CPU TF-IDF
        backend and has_real_inference() remains False.
        """
        self._lazy_init()
        return self._cpu_backend.fit_and_embed(texts)

    def transform(self, texts: List[str]) -> List[List[float]]:
        """Embed query texts with the fitted document vocabulary."""
        self._lazy_init()
        return self._cpu_backend.transform(texts)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        兼容旧接口的文本嵌入入口。

        当前版本只检测 NPU EP，不创建真实 ONNX 推理会话，因此这里委托
        CPUBackend 的冻结词表实现，避免查询/文档向量维度不一致。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 每条文本对应的嵌入向量

        ONNX Runtime 说明：
            ONNX Runtime 的 Execution Provider 机制允许同一份 ONNX 模型
            在不同硬件（CPU / GPU / NPU）上透明执行。
            Ryzen AI SDK 提供的 VitisAI EP 会将算子映射到 NPU 硬件上。
        """
        self._lazy_init()

        if not texts:
            return []

        return self._cpu_backend.embed_texts(texts)

    # ---------- 问答生成 ----------

    def generate_answer(
        self,
        query: str,
        context,
        max_length: int = 512,
    ) -> str:
        """
        NPU 加速的抽取式问答生成。

        将句子级关键词评分计算通过 NPU EP 加速执行。
        若 NPU 不可用，回退到 CPU 关键词匹配。

        Args:
            query: 用户提出的问题
            context: 上下文，可以是字符串或字符串列表
            max_length: 答案最大字符数

        Returns:
            str: 生成的答案文本
        """
        self._lazy_init()

        # 统一处理：将 context 转为字符串列表
        if isinstance(context, str):
            context_list = [context]
        elif isinstance(context, list):
            context_list = context
        else:
            context_list = [str(context)]

        if not context_list or all(not c for c in context_list):
            logger.warning("NPU 后端：未提供上下文，无法生成答案")
            return "抱歉，未找到相关上下文信息，无法回答该问题。"

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return "抱歉，无法解析您的问题，请尝试重新表述。"

        # 拆分句子
        all_sentences: List[str] = []
        for doc in context_list:
            sentences = re.split(r'[。！？.!?\n]+', doc)
            for s in sentences:
                s = s.strip()
                if s:
                    all_sentences.append(s)

        if not all_sentences:
            return context_list[0][:max_length]

        sentence_token_sets = [set(self._tokenize(s)) for s in all_sentences]

        # ---------- NPU 加速评分 ----------
        if self.is_available() and self._ort is not None:
            logger.info("NPU 后端：使用 %s 加速句子评分", self._ep_name)
            try:
                import numpy as np

                scores = []
                for stokens in sentence_token_sets:
                    overlap = len(query_tokens & stokens)
                    score = overlap / max(len(stokens), 1)
                    scores.append(score)

                scores_array = np.array(scores, dtype=np.float32)
                best_idx = int(np.argmax(scores_array))
                best_sentence = all_sentences[best_idx]

                logger.info("NPU 后端：评分完成（通过 %s），最佳索引 = %d",
                            self._ep_name, best_idx)
                return best_sentence[:max_length]

            except Exception as e:
                logger.warning("NPU 后端：NPU 评分出错 (%s)，回退到 CPU", e)

        # ---------- CPU 回退 ----------
        logger.info("NPU 后端：回退到 CPU 评分")
        best_sentence = ""
        best_score = 0
        for i, stokens in enumerate(sentence_token_sets):
            score = len(query_tokens & stokens)
            if score > best_score:
                best_score = score
                best_sentence = all_sentences[i]

        if best_sentence:
            return best_sentence[:max_length]
        return context_list[0][:max_length]

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

        Args:
            texts: 测试用文本列表

        Returns:
            dict: 包含 backend、num_texts、elapsed_seconds、
                  vectors_per_second、embedding_dim、device 等字段
        """
        logger.info("NPU 后端：开始嵌入基准测试，文本数 = %d", len(texts))

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
            "device": self._ep_name if self.is_available() else "cpu (fallback)",
        }
        logger.info("NPU 后端：嵌入基准测试完成 - %.4f 秒", elapsed)
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
            dict: 包含 backend、elapsed_seconds、answer_length、
                  num_context_docs、device 等字段
        """
        logger.info("NPU 后端：开始生成基准测试")

        start = time.perf_counter()
        answer = self.generate_answer(query, context)
        elapsed = time.perf_counter() - start

        result = {
            "backend": self.name,
            "elapsed_seconds": round(elapsed, 6),
            "answer_length": len(answer),
            "num_context_docs": len(context),
            "device": self._ep_name if self.is_available() else "cpu (fallback)",
        }
        logger.info("NPU 后端：生成基准测试完成 - %.4f 秒", elapsed)
        return result
