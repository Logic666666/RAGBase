"""
事件总线（发布-订阅）

Agent 步骤事件实时推送：
  Agent._trace → EventBus.publish → SSE endpoint → 前端 EventSource

接口抽象 + 单一实现：
  EventBus 是抽象接口，InMemoryEventBus 是当前实现（单机内存）。
  将来多进程部署替换为 Redis 实现，接口不变。
"""

from .bus import EventBus, InMemoryEventBus

__all__ = ["EventBus", "InMemoryEventBus"]
