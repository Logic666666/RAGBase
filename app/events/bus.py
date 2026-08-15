"""
事件总线（发布-订阅）

解决情境：研究报告长任务的过程不可见——轮询只能看到状态，
看不到 thought/工具调用逐步出现。EventBus 把 Agent 的每一步
实时推送给订阅者（SSE endpoint → 前端）。

设计（对齐 agent-event-bus 模式：hooks → event bus → SSE 广播）：
  Agent 每步 emit 事件 → EventBus 分发给订阅该 session 的队列
  → SSE endpoint 从队列读事件 → text/event-stream 推给前端

可替换原则：
  当前是进程内内存实现（单机够用）。
  将来多进程/多机器部署时，替换为 Redis 实现（跨进程广播），
  接口不变，上层零改动。
"""

import asyncio
import logging
from abc import ABC, abstractmethod


class EventBus(ABC):
    """事件总线抽象接口（实现可替换：内存 / Redis）"""

    @abstractmethod
    def subscribe(self, session_id: str) -> asyncio.Queue:
        """订阅某会话的事件流，返回该连接的队列"""
        ...

    @abstractmethod
    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """取消订阅（连接断开时）"""
        ...

    @abstractmethod
    async def publish(self, session_id: str, event: dict) -> None:
        """发布一个事件给该会话的所有订阅者"""
        ...


class InMemoryEventBus(EventBus):
    """
    进程内内存实现。

    每个订阅者一个队列（按 session_id 分组）。
    慢消费者保护：队列满时丢弃新事件（避免阻塞 Agent 主流程）。
    """

    def __init__(self, max_queue_size: int = 200):
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self.max_queue_size = max_queue_size

    def subscribe(self, session_id: str) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(session_id)
        if subscribers:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)

    async def publish(self, session_id: str, event: dict) -> None:
        for queue in list(self._subscribers.get(session_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # 慢消费者：丢弃该事件，避免阻塞 Agent
                logging.debug(f"事件队列已满，丢弃事件: {event.get('event')}")
