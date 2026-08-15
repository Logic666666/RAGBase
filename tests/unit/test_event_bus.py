"""
事件总线单元测试

验证：订阅/发布/退订、按会话隔离、慢消费者保护。
"""

import asyncio

import pytest

from app.events import InMemoryEventBus


@pytest.mark.asyncio
async def test_publish_reaches_subscriber():
    """订阅者收到发布的事件"""
    bus = InMemoryEventBus()
    q = bus.subscribe("session_a")

    await bus.publish("session_a", {"event": "tool_call", "detail": "search_kb"})
    event = await asyncio.wait_for(q.get(), timeout=1)

    assert event["event"] == "tool_call"
    assert event["detail"] == "search_kb"


@pytest.mark.asyncio
async def test_session_isolation():
    """事件按会话隔离：不同 session 互不干扰"""
    bus = InMemoryEventBus()
    qa = bus.subscribe("session_a")
    qb = bus.subscribe("session_b")

    await bus.publish("session_a", {"event": "plan", "detail": "计划A"})

    # A 收到，B 收不到
    event_a = await asyncio.wait_for(qa.get(), timeout=1)
    assert event_a["detail"] == "计划A"
    assert qb.empty()


@pytest.mark.asyncio
async def test_unsubscribe():
    """退订后不再收到事件"""
    bus = InMemoryEventBus()
    q = bus.subscribe("session_a")
    bus.unsubscribe("session_a", q)

    await bus.publish("session_a", {"event": "tool_call", "detail": "x"})
    assert q.empty()


@pytest.mark.asyncio
async def test_slow_consumer_not_blocking():
    """慢消费者（队列满）不阻塞发布者"""
    bus = InMemoryEventBus(max_queue_size=2)
    bus.subscribe("session_a")  # 订阅但不消费

    # 发布超过队列容量的事件，不应抛异常/阻塞
    for i in range(10):
        await bus.publish("session_a", {"event": "e", "detail": str(i)})
    # 发布者正常完成（队列丢弃了溢出的事件）


@pytest.mark.asyncio
async def test_agent_publishes_to_bus():
    """Agent 执行时事件发布到 EventBus（_trace 单一事件源）"""
    from app.events import InMemoryEventBus
    from app.agent.orchestrator import Agent
    from app.tools.base import BaseTool, ToolSpec
    from app.tools.registry import ToolRegistry
    from tests.unit.test_agent_orchestrator import ScriptedLLM

    # 构造 registry（mock 工具）
    class FakeKB(BaseTool):
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="search_kb",
                description="搜索知识库",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        async def run(self, query: str) -> str:
            return f"[文档] {query}"

    registry = ToolRegistry()
    registry.register(FakeKB())

    bus = InMemoryEventBus()
    q = bus.subscribe("test_session")

    llm = ScriptedLLM([
        '{"tool": "search_kb", "arguments": {"query": "数据库"}}',
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5,
                  event_bus=bus, session_id="test_session")
    await agent.run("调研")
    # create_task 调度的发布协程需要时间执行完
    await asyncio.sleep(0.05)

    # 收集发布的事件
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    event_types = [e["event"] for e in events]
    assert "plan" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
