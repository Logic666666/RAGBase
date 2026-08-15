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

    - chat()：规划轮固定返回 plan（Agent.run 开头调用）
    - chat_with_response()：ReAct 循环按 scripts 顺序返回
    两种调用分别记录（chat_calls / llm_calls），索引互不干扰。
    """

    def __init__(self, scripts: list[str], thinking: str = "",
                 plan: str = '{"plan": [{"dimension": "技术方案", "keywords": ["架构"]}]}'):
        self.scripts = scripts
        self.thinking = thinking  # 可选的思考内容，验证 think 事件传递
        self.plan = plan
        self.chat_calls: list = []   # 规划轮调用
        self.llm_calls: list = []    # ReAct 循环调用

    async def chat(self, messages):
        self.chat_calls.append(messages)
        return self.plan

    async def chat_with_response(self, messages):
        from app.infrastructure.llm_client import ChatResponse
        self.llm_calls.append(messages)
        idx = min(len(self.llm_calls) - 1, len(self.scripts) - 1)
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
    system(规则) → system(研究计划) → user(任务) → assistant(工具调用) → user(工具结果)
    """
    llm = ScriptedLLM([
        '{"tool": "search_kb", "arguments": {"query": "数据库"}}',
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    await agent.run("测试任务")

    history = llm.llm_calls[0]
    roles = [m.role for m in history]
    assert roles == ["system", "system", "system", "user", "assistant", "user"]

    # 第二条 system 是项目文件结构锚点（已探索，防重复 list_files）
    assert "项目文件结构" in history[1].content
    # 第三条 system 是研究计划锚点
    assert "研究计划" in history[2].content

    # 工具结果以 [工具结果 xxx] 前缀写回
    assert "[工具结果 search_kb]" in history[-1].content
    assert "关于 数据库 的内容" in history[-1].content


@pytest.mark.asyncio
async def test_plan_round_recorded(registry):
    """规划轮：研究计划进入消息历史 + trace 有 plan 事件"""
    from app.observability.tracer import Tracer

    llm = ScriptedLLM([
        '{"tool": "search_kb", "arguments": {"query": "数据库"}}',
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5, tracer=Tracer())
    await agent.run("调研数据库方案")

    # trace 有 plan 事件
    events = [e.event for e in agent.tracer.events]
    assert "plan" in events, f"缺少 plan 事件: {events}"
    # 规划轮单独调用了一次 chat
    assert len(llm.chat_calls) == 1


@pytest.mark.asyncio
async def test_plan_explores_structure(registry):
    """
    规划轮的探索使用统一的 tool_call/tool_result 事件
    （与主循环一致，不区分阶段——对齐 Claude Code）。
    """
    from app.observability.tracer import Tracer
    from app.tools.builtin.codebase import ListFilesTool

    # 注册 list_files 工具
    import tempfile, os
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "src"), exist_ok=True)
    registry.register(ListFilesTool(tmp))

    llm = ScriptedLLM([
        '{"tool": "search_kb", "arguments": {"query": "数据库"}}',
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5, tracer=Tracer())
    await agent.run("调研数据库方案")

    events = [e.event for e in agent.tracer.events]
    # 规划前的探索是统一工具调用：tool_call(list_files) 先于 plan
    assert "tool_call" in events
    first_tool_call = next(i for i, e in enumerate(agent.tracer.events) if e.event == "tool_call")
    plan_idx = events.index("plan")
    assert first_tool_call < plan_idx, "探索工具调用应先于 plan"
    # 工具调用 detail 明确展示工具名
    assert "list_files" in agent.tracer.events[first_tool_call].detail

    # 规划轮的消息里包含项目结构
    assert len(llm.chat_calls) == 1
    plan_prompt = llm.chat_calls[0][-1].content
    assert "项目文件结构" in plan_prompt


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
async def test_duplicate_tool_call_detected(registry):
    """
    收敛检测：相同 (工具, 参数) 重复调用 → 返回"重复调用"错误，
    不实际执行工具（防死循环/无进展烧 token）。
    """
    # thought 不同（输出不同，不触发响应级检测）但工具+参数相同
    llm = ScriptedLLM([
        '{"thought": "第一次搜索", "tool": "search_kb", "arguments": {"query": "数据库"}}',
        '{"thought": "再试一次", "tool": "search_kb", "arguments": {"query": "数据库"}}',
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("调研")

    # 重复调用返回错误提示，模型看到后换策略
    assert result.completed is True
    # 第二次循环调用时，消息历史里应有"重复调用"错误
    history = llm.llm_calls[1]
    dup_msg = next(m for m in history if m.role == "user" and m.content.startswith("[工具错误"))
    assert "重复调用" in dup_msg.content


@pytest.mark.asyncio
async def test_empty_tool_name_rejected(registry):
    """
    空工具名（模型生成退化：tool=""）应明确报错并计入失败计数，
    不执行空工具。
    """
    llm = ScriptedLLM([
        '{"thought": "调用工具", "tool": "", "arguments": {}}',  # 空 tool
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("调研")

    # 模型从错误中恢复，最终正常完成
    assert result.completed is True
    assert "回答完毕" in result.answer


@pytest.mark.asyncio
async def test_empty_tool_name_gives_up_after_limit(registry):
    """空工具名持续出现 → 达到失败上限终止（不无限循环）"""
    llm = ScriptedLLM([
        '{"tool": "", "arguments": {}}',
        '{"tool": "", "arguments": {}}',
        '{"tool": "", "arguments": {}}',
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5, max_parse_failures=3)
    result = await agent.run("调研")

    assert result.completed is False
    assert result.reason == "tool_errors"


@pytest.mark.asyncio
async def test_null_tool_name_rejected(registry):
    """
    模型输出 "tool": null（JSON null → "None"）→ 应被拦截，
    不执行名为 "None" 的空工具。
    """
    llm = ScriptedLLM([
        '{"thought": "调用工具", "tool": null, "arguments": {}}',  # null 工具名
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("调研")

    # 模型从错误中恢复，正常完成
    assert result.completed is True
    assert "回答完毕" in result.answer


@pytest.mark.asyncio
async def test_echo_mode_detected(registry):
    """
    响应级收敛检测：模型复读自身（每轮输出完全相同）→ 检测到无进展并终止。
    工具级重复检测拦不住"整段复读"（thought 也相同），需要响应级拦截。
    """
    echo_output = '{"tool": "list_files", "arguments": {"pattern": "*"}}'
    llm = ScriptedLLM([echo_output, echo_output, echo_output, echo_output])
    agent = Agent(llm=llm, tools=registry, max_steps=5, max_parse_failures=3)

    result = await agent.run("调研")

    # 复读被检测为无进展 → 未完成（连续无进展终止）
    assert result.completed is False
    assert "无进展" in result.answer
    assert result.reason == "tool_errors"


@pytest.mark.asyncio
async def test_different_args_not_duplicate(registry):
    """不同参数的相同工具调用不算重复"""
    llm = ScriptedLLM([
        '{"tool": "search_kb", "arguments": {"query": "数据库"}}',
        '{"tool": "search_kb", "arguments": {"query": "向量数据库"}}',  # 不同 query
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("调研")

    # 两次都正常执行（无重复错误）
    assert result.completed is True
    history = llm.llm_calls[1]
    assert not any("重复调用" in m.content for m in history if m.role == "user")


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

    # 连续相同输出被响应级收敛检测拦截（无进展）或参数错误，
    # 最终达到阈值放弃——循环不无限跑
    assert result.completed is False
    assert result.reason == "tool_errors"
