"""
LocalDoc Agent - 答案生成器模块

根据检索到的文档块和用户查询生成回答。

工作模式：
1. 后端模式：使用外部 LLM 后端（如 NPU/GPU 上的模型）生成自然语言回答
2. 回退模式：使用抽取式方法，从检索到的文档中提取最相关的句子作为回答

格式化上下文：将检索到的文档块格式化为 LLM 可理解的 prompt 上下文。
"""

from typing import Optional

from localdoc.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerGenerator:
    """
    答案生成器 - 根据查询和上下文生成回答。

    支持两种工作模式：
    1. 后端模式：调用 LLM 后端的 generate_answer 方法
    2. 抽取式回退：从上下文中提取最相关的句子，拼接为回答

    Args:
        backend: LLM 推理后端（可选），需实现
                 generate_answer(query, context) -> str 方法

    用法示例：
        generator = AnswerGenerator(backend=llm_backend)
        answer = generator.generate("什么是异构计算?", context_chunks)
    """

    def __init__(self, backend=None) -> None:
        self.backend = backend

        if backend is not None:
            logger.info(
                f"AnswerGenerator 使用后端模式: {type(backend).__name__}"
            )
        else:
            logger.info("AnswerGenerator 使用抽取式回退模式")

    def generate(self, query: str, context_chunks: list) -> str:
        """
        根据查询和上下文块生成回答。

        Args:
            query: 用户查询
            context_chunks: 检索到的相关文档块列表，每个元素为字典，
                          需包含 'content' 和可选的 'score' 字段

        Returns:
            生成的回答文本
        """
        if not context_chunks:
            return "抱歉，未找到与问题相关的文档内容。请确认已加载相关文档。"

        if self.backend is not None:
            return self._generate_with_backend(query, context_chunks)
        else:
            return self._generate_extractive(query, context_chunks)

    def _generate_with_backend(self, query: str, context_chunks: list) -> str:
        """
        使用 LLM 后端生成回答。

        Args:
            query: 用户查询
            context_chunks: 上下文文档块

        Returns:
            LLM 生成的回答
        """
        # 将上下文块格式化为 prompt
        context = self.format_context(context_chunks)

        try:
            answer = self.backend.generate_answer(query=query, context=context)
            logger.info(f"后端生成回答成功 ({len(answer)} 字符)")
            return answer
        except Exception as e:
            logger.error(
                f"后端生成失败，回退到抽取式: {type(e).__name__}: {e}"
            )
            return self._generate_extractive(query, context_chunks)

    def _generate_extractive(self, query: str, context_chunks: list) -> str:
        """
        抽取式回答生成（无需 LLM 的回退方案）。

        从上下文中提取与查询最相关的句子，按相关性排序后拼接为回答。

        Args:
            query: 用户查询
            context_chunks: 上下文文档块

        Returns:
            抽取式生成的回答
        """
        import re

        # 从查询中提取关键词（简单的分词处理）
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            # 无法提取关键词时，直接返回得分最高的文档块内容
            if context_chunks:
                top_chunk = context_chunks[0]
                return (
                    f"根据文档内容，找到以下相关信息：\n\n"
                    f"{top_chunk['content'][:500]}"
                )
            return "未能找到相关信息。"

        # 从所有上下文块中提取句子并评分
        scored_sentences: list[tuple[float, str, str]] = []

        for chunk in context_chunks:
            content = chunk.get("content", "")
            source = chunk.get("source", "")
            chunk_score = chunk.get("score", 0.5)  # 块的相关性分数

            # 按句子分割
            sentences = re.split(r'(?<=[。！？.!?])\s*', content)
            sentences = [s.strip() for s in sentences if s.strip()]

            for sentence in sentences:
                if len(sentence) < 5:
                    continue  # 跳过过短的句子

                # 计算句子的相关性分数
                sentence_score = self._score_sentence(
                    sentence, query_keywords, chunk_score
                )
                scored_sentences.append((sentence_score, sentence, source))

        # 按分数降序排序
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # 取得分最高的句子（最多 5 句）
        top_sentences = scored_sentences[:5]

        if not top_sentences:
            return "抱歉，未能从文档中提取到与问题直接相关的内容。"

        # 拼接为回答
        answer_parts = ["根据文档内容，找到以下相关信息：\n"]
        for i, (score, sentence, source) in enumerate(top_sentences, 1):
            source_info = f"（来源: {source}）" if source else ""
            answer_parts.append(f"{i}. {sentence}{source_info}")

        answer = "\n".join(answer_parts)
        logger.info(
            f"抽取式回答生成完成: 提取 {len(top_sentences)} 个相关句子"
        )
        return answer

    def _extract_keywords(self, text: str) -> list:
        """
        从文本中提取关键词。

        简单实现：使用正则提取中文字符序列和英文单词，
        过滤掉常见的停用词。

        Args:
            text: 输入文本

        Returns:
            关键词列表
        """
        import re

        # 提取中文词（2字以上）和英文单词
        chinese_words = re.findall(r'[一-鿿]{2,}', text)
        english_words = [
            w.lower() for w in re.findall(r'[a-zA-Z]{2,}', text)
        ]

        # 简单的停用词过滤
        stop_words = {
            "什么", "如何", "怎么", "为什么", "哪些", "请问",
            "可以", "能够", "应该", "需要", "已经", "是否",
            "the", "is", "are", "was", "were", "how", "what",
            "where", "when", "which", "this", "that",
        }

        keywords = [
            w for w in chinese_words + english_words
            if w not in stop_words
        ]

        return keywords

    def _score_sentence(
        self,
        sentence: str,
        query_keywords: list,
        chunk_score: float,
    ) -> float:
        """
        计算句子与查询的相关性分数。

        评分依据：
        1. 关键词命中率：句子中包含多少查询关键词
        2. 块相关性：句子所在文档块的检索分数
        3. 句子长度惩罚：避免过短的片段

        Args:
            sentence: 待评分的句子
            query_keywords: 查询关键词列表
            chunk_score: 句子所在文档块的检索相关性分数

        Returns:
            相关性分数（越高越相关）
        """
        if not query_keywords:
            return chunk_score

        # 计算关键词命中数
        hit_count = sum(
            1 for kw in query_keywords if kw in sentence.lower()
        )

        # 命中率
        hit_ratio = hit_count / len(query_keywords)

        # 长度因子：偏好中等长度的句子（20-200 字符）
        length = len(sentence)
        if length < 20:
            length_factor = 0.5
        elif length > 200:
            length_factor = 0.8
        else:
            length_factor = 1.0

        # 综合分数 = 关键词命中率 * 0.6 + 块相关性 * 0.3 + 长度因子 * 0.1
        final_score = (
            hit_ratio * 0.6 + chunk_score * 0.3 + length_factor * 0.1
        )

        return final_score

    @staticmethod
    def format_context(chunks: list) -> str:
        """
        将检索到的文档块格式化为 LLM 可理解的上下文字符串。

        格式：
        --- 文档 1 (来源: xxx) ---
        内容...

        --- 文档 2 (来源: yyy) ---
        内容...

        Args:
            chunks: 文档块列表

        Returns:
            格式化后的上下文字符串
        """
        if not chunks:
            return ""

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            source = chunk.get("source", "未知来源")
            score = chunk.get("score", None)

            header = f"--- 文档 {i} (来源: {source})"
            if score is not None:
                header += f", 相关度: {score:.2f}"
            header += " ---"

            parts.append(f"{header}\n{content}")

        return "\n\n".join(parts)
