"""
LocalDoc Agent - 文档加载器模块

负责从各种格式的文档中提取文本内容。
支持 Markdown (.md)、纯文本 (.txt)、PDF (.pdf) 等格式。

使用 pathlib 进行路径操作，对所有 I/O 错误进行优雅处理。
"""

from pathlib import Path
from typing import Optional

from localdoc.utils.logger import get_logger

logger = get_logger(__name__)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


class DocumentLoader:
    """
    文档加载器 - 负责读取各类文档并提取纯文本内容。

    支持的文件格式：
    - Markdown (.md)：直接读取原始文本
    - 纯文本 (.txt)：直接读取原始文本
    - PDF (.pdf)：使用 PyPDF2 或 pdfplumber 提取文本（可选依赖）

    用法示例：
        loader = DocumentLoader()
        content = loader.load_markdown("example.md")
        docs = loader.load_directory("./docs/")
    """

    def load_markdown(self, file_path: str) -> str:
        """
        加载 Markdown 文件并返回文本内容。

        Args:
            file_path: Markdown 文件路径

        Returns:
            文件的文本内容

        Raises:
            FileNotFoundError: 文件不存在
            IOError: 文件读取失败
        """
        path = Path(file_path)
        self._validate_file(path, ".md")

        try:
            content = path.read_text(encoding="utf-8")
            logger.info(f"成功加载 Markdown 文件: {path.name} ({len(content)} 字符)")
            return content
        except UnicodeDecodeError:
            # 回退到 GBK 编码（部分中文文档可能使用 GBK）
            logger.warning(f"UTF-8 解码失败，尝试 GBK 编码: {path.name}")
            return path.read_text(encoding="gbk")

    def load_text(self, file_path: str) -> str:
        """
        加载纯文本文件并返回文本内容。

        Args:
            file_path: 文本文件路径

        Returns:
            文件的文本内容

        Raises:
            FileNotFoundError: 文件不存在
            IOError: 文件读取失败
        """
        path = Path(file_path)
        self._validate_file(path, ".txt")

        try:
            content = path.read_text(encoding="utf-8")
            logger.info(f"成功加载文本文件: {path.name} ({len(content)} 字符)")
            return content
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 解码失败，尝试 GBK 编码: {path.name}")
            return path.read_text(encoding="gbk")

    def load_pdf(self, file_path: str) -> str:
        """
        加载 PDF 文件并提取文本内容。

        优先使用 pdfplumber（提取效果更好），
        如果未安装则回退到 PyPDF2。
        两者均未安装时抛出 ImportError。

        Args:
            file_path: PDF 文件路径

        Returns:
            从 PDF 中提取的文本内容

        Raises:
            FileNotFoundError: 文件不存在
            ImportError: PyPDF2 和 pdfplumber 均未安装
        """
        path = Path(file_path)
        self._validate_file(path, ".pdf")

        # 优先尝试 pdfplumber（对中文支持更好）
        try:
            import pdfplumber

            return self._extract_with_pdfplumber(path)
        except ImportError:
            logger.info("pdfplumber 未安装，尝试使用 PyPDF2")

        # 回退到 PyPDF2
        try:
            from PyPDF2 import PdfReader

            return self._extract_with_pypdf2(path)
        except ImportError:
            raise ImportError(
                "需要安装 PDF 处理库才能加载 PDF 文件。"
                "请执行: pip install pdfplumber  或  pip install PyPDF2"
            )

    def _extract_with_pdfplumber(self, path: Path) -> str:
        """使用 pdfplumber 提取 PDF 文本。"""
        import pdfplumber

        pages_text: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text)
            logger.info(
                f"pdfplumber 成功提取 PDF: {path.name} "
                f"(共 {len(pdf.pages)} 页, {len(pages_text)} 页有文本)"
            )
        return "\n\n".join(pages_text)

    def _extract_with_pypdf2(self, path: Path) -> str:
        """使用 PyPDF2 提取 PDF 文本。"""
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        pages_text: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)
        logger.info(
            f"PyPDF2 成功提取 PDF: {path.name} "
            f"(共 {len(reader.pages)} 页, {len(pages_text)} 页有文本)"
        )
        return "\n\n".join(pages_text)

    def load_file(self, file_path: str) -> dict:
        """
        根据文件扩展名自动选择加载方式，返回内容和元数据。

        Args:
            file_path: 文件路径

        Returns:
            字典，包含:
            - 'content' (str): 文本内容
            - 'source' (str): 文件路径字符串
            - 'type' (str): 文件类型（扩展名）
            - 'size' (int): 文件大小（字节）
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if not path.is_file():
            logger.error(f"路径不是文件: {file_path}")
            raise ValueError(f"路径不是文件: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"不支持的文件类型 '{ext}'，将尝试作为文本文件加载")
            ext = ".txt"  # 未知类型尝试作为文本处理

        # 根据扩展名分发到对应的加载方法
        loader_map = {
            ".md": self.load_markdown,
            ".txt": self.load_text,
            ".pdf": self.load_pdf,
        }

        try:
            loader_func = loader_map.get(ext, self.load_text)
            content = loader_func(file_path)
        except ImportError:
            # PDF 库缺失时记录错误但不中断
            logger.error(f"无法加载 PDF 文件（缺少依赖库）: {path.name}")
            raise
        except Exception as e:
            logger.error(f"加载文件失败 [{path.name}]: {type(e).__name__}: {e}")
            raise

        return {
            "content": content,
            "source": str(path.resolve()),
            "type": ext.lstrip("."),
            "size": path.stat().st_size,
        }

    def load_directory(self, dir_path: str) -> list:
        """
        递归加载目录下所有支持格式的文件。

        Args:
            dir_path: 目录路径

        Returns:
            列表，每个元素为字典:
            - 'content' (str): 文件文本内容
            - 'source' (str): 文件绝对路径
            - 'type' (str): 文件类型

        Raises:
            FileNotFoundError: 目录不存在
        """
        path = Path(dir_path)

        if not path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        if not path.is_dir():
            raise ValueError(f"路径不是目录: {dir_path}")

        documents: list = []
        # 按扩展名筛选文件，忽略隐藏文件和以 ~ 开头的临时文件
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in sorted(path.rglob(f"*{ext}")):
                # 跳过隐藏文件和临时文件
                if file_path.name.startswith(".") or file_path.name.startswith("~"):
                    continue

                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    # 单个文件失败不应阻断整个目录的加载
                    logger.warning(
                        f"跳过无法加载的文件 [{file_path.name}]: "
                        f"{type(e).__name__}: {e}"
                    )
                    continue

        logger.info(
            f"目录加载完成: {dir_path} - "
            f"成功 {len(documents)} 个文件"
        )
        return documents

    def _validate_file(self, path: Path, expected_ext: str) -> None:
        """
        验证文件路径的有效性。

        Args:
            path: 文件路径对象
            expected_ext: 期望的文件扩展名

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件扩展名不匹配
        """
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if not path.is_file():
            raise ValueError(f"路径不是文件: {path}")

        actual_ext = path.suffix.lower()
        if actual_ext != expected_ext:
            logger.warning(
                f"文件扩展名不匹配: 期望 '{expected_ext}'，实际 '{actual_ext}'"
            )
