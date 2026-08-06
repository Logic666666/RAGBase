"""
可靠性层（对齐 Claude Code 的 error recovery 设计）

解决情境：长任务（研究报告）中途网络闪断/LLM 超时/服务限流，
一次 transient 错误不应导致整个任务 failed。

组件：
  errors.py  错误分类（transient 可重试 / permanent 不可重试）
  retry.py   重试策略（指数退避 + 抖动，限次防烧 token）

设计原则（接口抽象 + 单一实现）：
  RetryPolicy 是抽象接口，ExponentialBackoff 是当前实现。
  上层（llm_client）只依赖接口，将来换策略只加实现类。
"""

from .errors import ErrorCategory, classify_error
from .retry import ExponentialBackoff, RetryPolicy

__all__ = ["ErrorCategory", "classify_error", "RetryPolicy", "ExponentialBackoff"]
