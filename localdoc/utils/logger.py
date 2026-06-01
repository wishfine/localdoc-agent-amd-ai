"""
LocalDoc Agent - 统一日志模块

提供项目统一的日志配置和获取接口。
所有模块通过 get_logger(name) 获取 logger 实例。
"""

import logging
import sys

# 日志格式：时间 | 级别 | 模块名 | 消息
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_LEVEL = logging.INFO

# 标记是否已经完成根日志器的配置
_root_configured = False


def _configure_root() -> None:
    """配置根日志器，仅执行一次。"""
    global _root_configured
    if _root_configured:
        return

    root_logger = logging.getLogger("localdoc")
    root_logger.setLevel(_DEFAULT_LEVEL)

    # 避免重复添加 handler
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(_DEFAULT_LEVEL)
        formatter = logging.Formatter(_DEFAULT_FORMAT)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    获取一个命名的 logger 实例。

    所有 logger 均挂载在 "localdoc" 根节点下，
    便于统一控制日志级别和输出格式。

    Args:
        name: logger 名称，通常为模块的 __name__ 或短名称
              例如 "localdoc.backends.cpu_backend"

    Returns:
        配置好的 logging.Logger 实例

    Usage:
        from localdoc.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("CPU 后端初始化完成")
    """
    _configure_root()

    # 如果传入的 name 已经带有 "localdoc." 前缀，直接使用
    # 否则自动加上前缀，保持层级结构
    if name.startswith("localdoc."):
        full_name = name
    elif name == "localdoc":
        full_name = "localdoc"
    else:
        full_name = f"localdoc.{name}"

    return logging.getLogger(full_name)


def set_level(level: int) -> None:
    """
    动态调整 localdoc 命名空间下所有 logger 的日志级别。

    Args:
        level: logging.DEBUG, logging.INFO, logging.WARNING 等
    """
    logger = logging.getLogger("localdoc")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
