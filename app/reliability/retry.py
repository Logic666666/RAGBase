"""
重试策略（对齐 Claude Code 的 retry with backoff）

Claude Code 对可恢复错误使用退避重试，且限制单轮重试次数
（max_output_tokens recovery 上限 3 次）——防止无限重试烧 token。

我们提供：
  RetryPolicy          抽象接口（实现可替换）
  ExponentialBackoff   指数退避 + 抖动
"""

import random
from abc import ABC, abstractmethod

from .errors import ErrorCategory


class RetryPolicy(ABC):
    """
    重试策略抽象接口。

    上层（llm_client）只依赖本接口：
      should_retry：这次要不要重试
      next_delay：  下次重试等多久

    将来换策略（如线性退避、固定间隔）只新增实现类。
    """

    @abstractmethod
    def should_retry(self, attempt: int, category: ErrorCategory) -> bool:
        """
        判断是否重试。

        Args:
            attempt: 已失败的次数（0 = 第一次失败）
            category: 错误类别（transient 才考虑重试）
        """
        ...

    @abstractmethod
    def next_delay(self, attempt: int) -> float:
        """下一次重试前的等待秒数"""
        ...


class ExponentialBackoff(RetryPolicy):
    """
    指数退避 + 抖动（目前方案）。

    延迟序列：base_delay * 2^attempt * random(0.8, 1.2)
    上限 max_delay 防无限增长；max_attempts 限总重试次数。
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
    ):
        """
        Args:
            max_attempts: 最大重试次数（对齐 Claude Code 的单轮 3 次上限）
            base_delay:   初始退避基数（秒）
            max_delay:    退避上限（秒）
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    def should_retry(self, attempt: int, category: ErrorCategory) -> bool:
        # 只有 transient 错误且未超过次数上限才重试
        if category != ErrorCategory.TRANSIENT:
            return False
        return attempt < self.max_attempts

    def next_delay(self, attempt: int) -> float:
        # 指数退避 + 抖动，最后封顶：
        # 先乘抖动再封顶，保证延迟严格不超过 max_delay
        delay = self.base_delay * (2 ** attempt) * random.uniform(0.8, 1.2)
        return min(self.max_delay, delay)
