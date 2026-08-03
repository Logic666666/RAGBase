"""
Agent 循环单元测试

用 mock LLM 验证 ReAct 循环的控制逻辑：
  - 正常完成（工具调用 → 观察 → 回答）
  - 消息历史结构
  - max_steps 截断（死循环保护）
  - 工具错误容忍
"""

import pytest

from app.agent.orchestrator import Agent
from app.tools.base import BaseTool, ToolSpec
from app.tools.registry import ToolRegistry


class FakeKB(BaseTool):
    """mock 的知识库工具"""

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
        return f"[文档] 关于 {query} 的内容"


class ScriptedLLM:
    """
    按脚本输出回复的 mock LLM。

    scripts: 每次调用按顺序返回对应文本。
    同时实现 chat 与 chat_with_response（orchestrator 使用后者）。
    """

    def __init__(self, scripts: list[str], thinking: str = ""):
        self.scripts = scripts
        self.calls: list = []
        self.thinking = thinking  # 可选的思考内容，验证 think 事件传递

    async def chat(self, messages):
        self.calls.append(messages)
        return self.scripts[min(len(self.calls) - 1, len(self.scripts) - 1)]

    async def chat_with_response(self, messages):
        from app.infrastructure.llm_client import ChatResponse
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.scripts) - 1)
        return ChatResponse(
            content=self.scripts[idx],
            thinking=self.thinking if idx == 0 else "",
        )


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(FakeKB())
    return r


@pytest.mark.asyncio
async def test_react_loop_completes(registry):
    """
    正常流程：LLM 先调工具，再回答。
    期望：2 步完成，answer 正确。
    """
    llm = ScriptedLLM([
        '{"thought": "先搜索", "tool": "search_kb", "arguments": {"query": "数据库"}}',
        "推荐使用 ChromaDB。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)

    result = await agent.run("对比数据库方案")

    assert result.completed is True
    assert result.steps == 2
    assert "ChromaDB" in result.answer


@pytest.mark.asyncio
async def test_message_history_structure(registry):
    """
    消息历史应包含：
    system → user(任务) → assistant(工具调用) → user(工具结果)
    """
    llm = ScriptedLLM([
        '{"tool": "search_kb", "arguments": {"query": "数据库"}}',
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    await agent.run("测试任务")

    history = llm.calls[1]
    roles = [m.role for m in history]
    assert roles == ["system", "user", "assistant", "user"]

    # 工具结果以 [工具结果 xxx] 前缀写回
    assert "[工具结果 search_kb]" in history[-1].content
    assert "关于 数据库 的内容" in history[-1].content


@pytest.mark.asyncio
async def test_max_steps_truncation(registry):
    """模型持续调用工具（死循环）→ max_steps 截断"""
    llm = ScriptedLLM(['{"tool": "search_kb", "arguments": {"query": "x"}}'])
    agent = Agent(llm=llm, tools=registry, max_steps=3)

    result = await agent.run("任务")

    assert result.completed is False
    assert result.steps == 3
    assert "最大步数" in result.answer


@pytest.mark.asyncio
async def test_think_event_recorded(registry):
    """
    模型的思考内容（ChatResponse.thinking）应记录为 trace 的 think 事件。
    """
    llm = ScriptedLLM(
        ['{"tool": "search_kb", "arguments": {"query": "数据库"}}', "回答完毕。"],
        thinking="我需要先分析一下任务",
    )
    from app.observability.tracer import Tracer
    agent = Agent(llm=llm, tools=registry, max_steps=5, tracer=Tracer())
    await agent.run("测试任务")

    events = [e.event for e in agent.tracer.events]
    assert "think" in events, f"缺少 think 事件: {events}"

    think_event = next(e for e in agent.tracer.events if e.event == "think")
    assert "我需要先分析一下任务" in think_event.detail


@pytest.mark.asyncio
async def test_invalid_params_tolerance(registry):
    """
    模型持续传错参数 → 连续错误达到阈值后放弃。
    验证循环不会无限跑下去。
    """
    llm = ScriptedLLM([
        '{"tool": "search_kb", "arguments": {}}',  # 缺必填参数
        '{"tool": "search_kb", "arguments": {}}',
        '{"tool": "search_kb", "arguments": {}}',
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5, max_parse_failures=2)

    result = await agent.run("任务")

    assert result.completed is False
    assert "连续出错" in result.answer or "工具" in result.answer
