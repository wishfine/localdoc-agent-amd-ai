"""
LocalDoc Agent - 文本分块器模块

将长文本切分为适合向量检索的较小文本块（chunk）。
采用"先段落、后句子"的两层切分策略，尽量保持语义完整性。

分块策略：
1. 优先按段落（空行）切分
2. 如果单个段落超过 chunk_size，再按句子切分
3. 相邻块之间保留 chunk_overlap 个字符的重叠，避免上下文断裂
"""

import re
from typing import Optional

from localdoc.utils.logger import get_logger

logger = get_logger(__name__)

# 中英文句子结束标记的正则表达式
# 匹配中文句号、问号、感叹号，以及英文句号、问号、感叹号后跟空白或结尾
_SENTENCE_PATTERN = re.compile(
    r'(?<=[。！？.!?])\s*'  # 句子分隔符
)


class TextChunker:
    """
    文本分块器 - 将长文本切分为固定大小的文本块。

    切分策略说明：
    - 首先按段落（双换行符）分割
    - 如果段落长度超过 chunk_size，进一步按句子分割
    - 如果单个句子仍然超过 chunk_size，则强制按字符数切割
    - 相邻块之间保留 chunk_overlap 个字符的重叠窗口

    Args:
        chunk_size: 每个块的最大字符数（默认 500）
        chunk_overlap: 相邻块之间的重叠字符数（默认 50）

    用法示例：
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_text(long_text, source="doc.md")
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正整数，当前值: {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap 不能为负数，当前值: {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.debug(
            f"TextChunker 初始化: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def chunk_text(self, text: str, source: str = "") -> list:
        """
        将文本切分为多个文本块。

        每个块是一个字典，包含以下字段：
        - content (str): 块的文本内容
        - source (str): 来源标识（如文件路径）
        - index (int): 块在文档中的序号（从 0 开始）
        - start_pos (int): 块在原始文本中的起始字符位置

        Args:
            text: 待切分的文本
            source: 文本来源标识

        Returns:
            文本块列表，每个元素为字典
        """
        if not text or not text.strip():
            logger.warning(f"空文本，无法分块 (source={source})")
            return []

        # 文本预处理：规范化空白字符
        text = text.strip()

        # 第一层：按段落切分
        paragraphs = self._split_paragraphs(text)

        # 第二层：对过长的段落按句子切分
        raw_chunks: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                raw_chunks.append(paragraph)
            else:
                # 段落过长，按句子进一步切分
                sentence_chunks = self._split_sentences(paragraph)
                raw_chunks.extend(sentence_chunks)

        # 应用重叠窗口并构建最终的块
        chunks = self._apply_overlap(raw_chunks, source)

        logger.info(
            f"文本分块完成: source='{source}', "
            f"总长度={len(text)}, 生成 {len(chunks)} 个块"
        )
        return chunks

    def _split_paragraphs(self, text: str) -> list:
        """
        按段落切分文本。

        段落之间的分隔符是一个或多个连续空行。
        单个换行符不会触发分段（可能是同一段落内的换行）。

        Args:
            text: 原始文本

        Returns:
            段落列表（已去除首尾空白）
        """
        # 按双换行符分割（匹配段落间的空行）
        paragraphs = re.split(r'\n\s*\n', text)
        # 过滤掉空白段落
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> list:
        """
        将过长的文本按句子切分，保证每个块不超过 chunk_size。

        如果单个句子仍然超过 chunk_size，则强制按字符数切割。

        Args:
            text: 待切分的文本

        Returns:
            文本块列表
        """
        # 按句子结束标记分割
        sentences = _SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current_chunk = ""

        for sentence in sentences:
            # 单个句子超长：需要强制切割
            if len(sentence) > self.chunk_size:
                # 先把当前累积的块保存
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # 对超长句子进行强制字符级切割
                forced_chunks = self._force_split(sentence)
                chunks.extend(forced_chunks)
                continue

            # 尝试将当前句子追加到当前块
            candidate = (
                f"{current_chunk}{sentence}" if current_chunk else sentence
            )

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        # 别忘了最后一个块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _force_split(self, text: str) -> list:
        """
        对超过 chunk_size 的文本进行强制字符级切割。

        尽量在标点符号或空格处断开，避免截断词语。

        Args:
            text: 超长文本

        Returns:
            切割后的文本块列表
        """
        chunks: list[str] = []
        remaining = text

        while len(remaining) > self.chunk_size:
            # 在 chunk_size 范围内寻找最佳断点
            cut_pos = self._find_break_point(remaining, self.chunk_size)
            chunks.append(remaining[:cut_pos].strip())
            remaining = remaining[cut_pos:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    def _find_break_point(self, text: str, max_pos: int) -> int:
        """
        在 max_pos 附近寻找最佳断点位置。

        优先在标点符号处断开，其次在空格处，最后硬切。

        Args:
            text: 原始文本
            max_pos: 最大允许的切割位置

        Returns:
            最佳切割位置
        """
        if max_pos >= len(text):
            return len(text)

        # 在 max_pos 之前向后搜索标点符号或空格
        search_start = max(0, max_pos - 50)  # 在最后 50 个字符范围内寻找

        # 优先在标点处断开
        for i in range(max_pos, search_start, -1):
            if text[i - 1] in "。！？.!?\n":
                return i

        # 其次在空格处断开
        for i in range(max_pos, search_start, -1):
            if text[i - 1] in " \t":
                return i

        # 找不到好的断点，硬切
        return max_pos

    def _apply_overlap(self, chunks: list, source: str) -> list:
        """
        为文本块列表添加重叠窗口，并构建最终的块字典。

        每个块的开头会包含上一个块末尾的 chunk_overlap 个字符，
        以确保上下文连续性。

        Args:
            chunks: 原始文本块列表
            source: 来源标识

        Returns:
            带有元数据的块字典列表
        """
        if not chunks:
            return []

        result: list = []
        current_pos = 0  # 跟踪在原始文本中的位置（近似值）

        for i, chunk_content in enumerate(chunks):
            # 从第二个块开始，添加与前一个块的重叠部分
            if i > 0 and self.chunk_overlap > 0:
                prev_chunk = chunks[i - 1]
                # 取前一个块末尾的 overlap 个字符
                overlap_text = prev_chunk[-self.chunk_overlap:]
                # 合并重叠文本
                overlapped_content = overlap_text + chunk_content
                # 如果合并后超过 chunk_size，截断到 chunk_size
                if len(overlapped_content) > self.chunk_size:
                    overlapped_content = overlapped_content[:self.chunk_size]
                chunk_content = overlapped_content

            # Adjust start_pos: for i >= 1, the content starts with overlap
            # from the previous chunk, so the actual position is earlier
            actual_start = current_pos - self.chunk_overlap if i > 0 else current_pos

            result.append({
                "content": chunk_content,
                "source": source,
                "index": i,
                "start_pos": max(0, actual_start),
            })
            current_pos += len(chunks[i])  # 使用原始块长度更新位置

        return result

    def chunk_documents(self, documents: list) -> list:
        """
        批量对多个文档进行分块。

        Args:
            documents: 文档列表，每个文档为字典，包含 'content' 和 'source' 字段

        Returns:
            所有文档的分块结果合并列表
        """
        all_chunks: list = []
        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            chunks = self.chunk_text(content, source=source)
            all_chunks.extend(chunks)

        logger.info(
            f"批量分块完成: {len(documents)} 个文档, "
            f"共 {len(all_chunks)} 个块"
        )
        return all_chunks
