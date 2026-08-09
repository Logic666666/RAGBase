"""
会话数据模型与存储接口

SessionRecord：一次 Agent 会话的完整状态（可序列化为 JSON 落盘）。
  - 对齐 Claude Code 的 session transcript：消息历史（messages）随会话落盘，
    支持中断后恢复执行（resume）与全程追溯。
SessionStore：会话存储的抽象接口（实现可替换）。
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class SessionStatus(str, Enum):
    """会话状态机：running → done / failed"""
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class SessionRecord:
    """
    一次 Agent 会话的完整记录。

    - 提交时：status=running，携带任务参数
    - 执行中：messages 持续更新（transcript 落盘）
    - 完成时：status=done，写入 result/steps/trace
    - 失败时：status=failed，写入 error

    messages: Agent 循环的完整消息历史（[{"role", "content"}, ...]），
              即 Claude Code 的 transcript——用于恢复执行与追溯。
    """
    session_id: str
    kb: str
    task: str
    max_steps: int
    status: SessionStatus = SessionStatus.RUNNING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[dict] = field(default_factory=list)   # transcript（消息历史）
    result: str | None = None          # 最终回答
    completed: bool | None = None      # Agent 是否正常完成
    reason: str | None = None          # 未完成原因（"max_steps" / "tool_errors"）
    steps: int | None = None           # 执行步数
    trace: dict | None = None          # 完整 trace（可复盘）
    error: str | None = None           # 失败原因


class SessionStore(ABC):
    """
    会话存储的抽象接口。

    上层（SessionManager / API）只依赖本接口：
      create / update / get / list

    当前实现：JsonSessionStore
    计划未来实现：SqliteSessionStore / RedisSessionStore
    替换 = 新增实现类并注册，上层零改动。
    """

    @abstractmethod
    async def create(self, record: SessionRecord) -> None:
        """保存新会话记录"""
        ...

    @abstractmethod
    async def update(self, session_id: str, patch: dict) -> None:
        """
        部分更新会话记录（如 status/result/trace/messages）。
        patch 是字段名 → 新值的字典。
        """
        ...

    @abstractmethod
    async def get(self, session_id: str) -> SessionRecord | None:
        """按 session_id 获取会话记录；不存在返回 None"""
        ...

    @abstractmethod
    async def list(self) -> list[SessionRecord]:
        """列出所有会话记录（按创建时间倒序）"""
        ...
