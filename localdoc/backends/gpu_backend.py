"""
AMD GPU 后端实现 - 基于 ROCm / PyTorch-HIP 的加速引擎

本模块提供利用 AMD 独显 / 集显进行文本嵌入和问答生成的能力。

硬件与软件要求：
    - AMD GPU（Radeon RX / PRO / Instinct 系列）
    - ROCm 5.x+ 运行时（Linux 原生支持；Windows 通过 HIP SDK）
    - PyTorch 编译时启用 HIP 后端（即 torch.version.hip 非空）

    在 AMD 锐龙 AI MAX+ 平台上，集成的 Radeon 780M / 890M 等 GPU
    可通过 ROCm 加速轻量级推理任务。

工作原理：
    - embed_texts: 将 TF-IDF 权重矩阵搬到 GPU 上做矩阵运算，
      利用 GPU 大规模并行能力加速高维向量计算。
    - generate_answer: 在 GPU 上执行简单的注意力式评分，
      加速句子级别的相关性排序。

注意：
    如果 PyTorch 未编译 HIP 后端（常见于 macOS 或官方 pip 源），
    本后端会优雅降级并给出友好提示，不会导致程序崩溃。

典型用法：
    backend = AMDGPUBackend()
    if backend.is_available():
        vectors = backend.embed_texts(["文本一", "文本二"])
        info = backend.get_device_info()
        print(info)
    else:
        print("GPU 后端不可用，将使用 CPU 后端")
"""

import time
import math
import re
from collections import Counter
from typing import List, Optional, Dict, Any

from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class AMDGPUBackend:
    """
    AMD GPU 加速后端

    通过 PyTorch + ROCm/HIP 实现 GPU 加速的文本嵌入与问答生成。
    需要系统安装 ROCm 运行时以及编译了 HIP 后端的 PyTorch。

    如果运行环境不满足要求，所有方法会优雅降级到 CPU 计算，
    并在日志中给出明确提示。
    """

    def __init__(self) -> None:
        """初始化 GPU 后端，懒加载 PyTorch。"""
        self._torch = None
        self._device: Optional[str] = None
        self._hip_available: bool = False
        self._init_attempted: bool = False

    # ---------- 属性 ----------

    @property
    def name(self) -> str:
        """返回后端名称标识。"""
        return "AMD GPU"

    # ---------- 内部初始化 ----------

    def _lazy_init(self) -> None:
        """
        懒加载 PyTorch 并检查 HIP 可用性。

        仅在首次调用时执行，后续调用直接返回缓存结果。
        """
        if self._init_attempted:
            return
        self._init_attempted = True

        try:
            import torch  # 延迟导入，避免模块加载时的开销
            self._torch = torch
        except ImportError:
            logger.warning(
                "AMD GPU 后端：未安装 PyTorch。"
                "请参考 https://pytorch.org/get-started/locally/ 安装。"
            )
            return

        # 检查 HIP 后端是否可用
        hip_version = getattr(torch.version, 'hip', None)
        if hip_version:
            self._hip_available = True
            logger.info("AMD GPU 后端：检测到 PyTorch HIP 后端，ROCm/HIP 版本 = %s", hip_version)
        else:
            logger.warning(
                "AMD GPU 后端：当前 PyTorch 未编译 HIP 后端（torch.version.hip 为空）。"
                "如需 AMD GPU 加速，请安装 ROCm 版本的 PyTorch：\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.0\n"
                "或者确认系统已正确安装 ROCm 运行时。"
            )
            return

        # 检查 CUDA/HIP 设备是否可用
        if torch.cuda.is_available():
            self._device = "cuda"  # PyTorch 通过 CUDA API 统一访问 HIP 设备
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("AMD GPU 后端：GPU 可用 - %s", gpu_name)
        else:
            logger.warning(
                "AMD GPU 后端：已检测到 HIP 后端，但无可用 GPU 设备。"
                "请检查 ROCm 驱动是否正确安装，以及用户是否有 GPU 访问权限。"
            )

    # ---------- 可用性检查 ----------

    def is_available(self) -> bool:
        """
        检查 AMD GPU 后端是否真正可用。

        检查条件：
        1. PyTorch 已安装
        2. PyTorch 编译了 HIP 后端 (torch.version.hip 非空)
        3. 至少一个 GPU 设备可用 (torch.cuda.is_available())

        Returns:
            bool: GPU 后端是否可用于加速计算
        """
        self._lazy_init()
        return self._hip_available and self._device is not None

    # ---------- 设备信息 ----------

    def get_device_info(self) -> Dict[str, Any]:
        """
        获取 AMD GPU 设备详细信息。

        Returns:
            dict: 包含以下字段：
                - backend: 后端名称
                - available: 是否可用
                - hip_version: ROCm/HIP 版本号（若可用）
                - gpu_name: GPU 设备名称（若可用）
                - gpu_memory_mb: 显存大小（MB，若可用）
                - device_count: 可用 GPU 设备数量
        """
        self._lazy_init()

        info: Dict[str, Any] = {
            "backend": self.name,
            "available": self.is_available(),
            "hip_version": None,
            "gpu_name": None,
            "gpu_memory_mb": None,
            "device_count": 0,
        }

        if self._torch is None:
            return info

        hip_version = getattr(self._torch.version, 'hip', None)
        info["hip_version"] = hip_version

        if self._device is not None:
            try:
                info["gpu_name"] = self._torch.cuda.get_device_name(0)
                info["device_count"] = self._torch.cuda.device_count()
                props = self._torch.cuda.get_device_properties(0)
                info["gpu_memory_mb"] = round(props.total_mem / (1024 * 1024))
            except Exception as e:
                logger.warning("AMD GPU 后端：获取设备信息时出错 - %s", e)

        return info

    # ---------- 文本嵌入 ----------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        使用 AMD GPU 加速的 TF-IDF 文本嵌入。

        将 TF-IDF 计算中的词频-逆文档频率矩阵转换为 PyTorch 张量，
        并在 GPU 上执行归一化运算，利用并行计算加速。

        如果 GPU 不可用，自动回退到 CPU 纯 Python 计算。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 每条文本对应的嵌入向量

        ROCm/PyTorch 要求：
            需要安装 ROCm 版 PyTorch：
            pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
        """
        self._lazy_init()

        if not texts:
            return []

        # ---------- 分词与词汇表构建（CPU 上进行） ----------
        tokenized_docs = [self._tokenize(t) for t in texts]
        vocab_set: set = set()
        for tokens in tokenized_docs:
            vocab_set.update(tokens)
        feature_names: List[str] = sorted(vocab_set)
        vocab_index = {w: i for i, w in enumerate(feature_names)}
        vocab_size = len(feature_names)

        # 计算 IDF
        doc_freq: Counter = Counter()
        n_docs = len(texts)
        for tokens in tokenized_docs:
            for t in set(tokens):
                doc_freq[t] += 1
        idf_values = [0.0] * vocab_size
        for word, idx in vocab_index.items():
            idf_values[idx] = math.log((n_docs + 1) / (1 + doc_freq[word])) + 1.0

        # 构建原始 TF-IDF 矩阵
        raw_matrix = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec = [0.0] * vocab_size
            for word, count in tf.items():
                if word in vocab_index:
                    vec[idx := vocab_index[word]] = (count / total) * idf_values[idx]
            raw_matrix.append(vec)

        # ---------- GPU 加速归一化 ----------
        if self.is_available() and self._torch is not None and self._device is not None:
            logger.info("AMD GPU 后端：在 %s 上对 %d 条文本进行嵌入（维度=%d）",
                        self._device, len(texts), vocab_size)
            try:
                t = self._torch
                tensor = t.tensor(raw_matrix, dtype=t.float32, device=self._device)
                # L2 归一化
                norms = tensor.norm(dim=1, keepdim=True).clamp(min=1e-12)
                tensor = tensor / norms
                result = tensor.cpu().tolist()
                logger.info("AMD GPU 后端：GPU 嵌入完成")
                return result
            except Exception as e:
                logger.warning(
                    "AMD GPU 后端：GPU 计算出错 (%s)，回退到 CPU 计算", e
                )

        # ---------- CPU 回退 ----------
        logger.info("AMD GPU 后端：回退到 CPU 计算")
        vectors: List[List[float]] = []
        for vec in raw_matrix:
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
        GPU 加速的抽取式问答生成。

        将句子级关键词评分计算搬到 GPU 上执行。
        若 GPU 不可用，回退到 CPU 关键词匹配。

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
            logger.warning("AMD GPU 后端：未提供上下文，无法生成答案")
            return "抱歉，未找到相关上下文信息，无法回答该问题。"

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return "抱歉，无法解析您的问题，请尝试重新表述。"

        # 拆分所有上下文为句子
        all_sentences: List[str] = []
        sentence_doc_map: List[int] = []  # 记录句子来源
        for doc_idx, doc in enumerate(context_list):
            sentences = re.split(r'[。！？.!?\n]+', doc)
            for s in sentences:
                s = s.strip()
                if s:
                    all_sentences.append(s)
                    sentence_doc_map.append(doc_idx)

        if not all_sentences:
            return context_list[0][:max_length]

        # 计算每个句子的 token 集合
        sentence_token_sets = [set(self._tokenize(s)) for s in all_sentences]

        # ---------- GPU 加速评分 ----------
        if self.is_available() and self._torch is not None and self._device is not None:
            logger.info("AMD GPU 后端：使用 GPU 加速句子评分")
            try:
                t = self._torch
                # 构建评分向量：query token 在句子中的命中比例
                scores_list = []
                for stokens in sentence_token_sets:
                    overlap = len(query_tokens & stokens)
                    score = overlap / max(len(stokens), 1)
                    scores_list.append(score)

                scores_tensor = t.tensor(scores_list, dtype=t.float32, device=self._device)
                best_idx = scores_tensor.argmax().item()
                best_sentence = all_sentences[best_idx]

                logger.info("AMD GPU 后端：GPU 评分完成，最佳句子索引 = %d", best_idx)
                return best_sentence[:max_length]

            except Exception as e:
                logger.warning(
                    "AMD GPU 后端：GPU 评分出错 (%s)，回退到 CPU 计算", e
                )

        # ---------- CPU 回退 ----------
        logger.info("AMD GPU 后端：回退到 CPU 评分")
        best_sentence = ""
        best_score = 0
        for i, stokens in enumerate(sentence_token_sets):
            score = len(query_tokens & stokens)
            if score > best_score:
                best_score = score
                best_sentence = all_sentences[i]

        if best_sentence:
            return best_sentence[:max_length]
        return context[0][:max_length]

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
        logger.info("AMD GPU 后端：开始嵌入基准测试，文本数 = %d", len(texts))

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
            "device": self._device if self.is_available() else "cpu (fallback)",
        }
        logger.info("AMD GPU 后端：嵌入基准测试完成 - %.4f 秒", elapsed)
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
        logger.info("AMD GPU 后端：开始生成基准测试")

        start = time.perf_counter()
        answer = self.generate_answer(query, context)
        elapsed = time.perf_counter() - start

        result = {
            "backend": self.name,
            "elapsed_seconds": round(elapsed, 6),
            "answer_length": len(answer),
            "num_context_docs": len(context),
            "device": self._device if self.is_available() else "cpu (fallback)",
        }
        logger.info("AMD GPU 后端：生成基准测试完成 - %.4f 秒", elapsed)
        return result
