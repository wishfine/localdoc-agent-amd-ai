"""LocalDoc Agent - 面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体"""

__version__ = "0.1.0"


def get_agent_class():
    """Lazy import to avoid circular dependency."""
    from localdoc.agent import LocalDocAgent
    return LocalDocAgent


def get_scheduler_class():
    """Lazy import to avoid circular dependency."""
    from localdoc.scheduler import HeterogeneousScheduler
    return HeterogeneousScheduler
