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
async def test_empty_tool_with_thought_is_final_answer(registry):
    """
    语义容错：tool=null/空 但 thought 有内容 = 模型在"思考后准备结束"，
    应视为最终回答（返回 thought），而非报"工具名为空"错误。
    """
    llm = ScriptedLLM([
        '{"thought": "已有足够信息，无需再调用工具", "tool": null, "arguments": {}}',
        '{"thought": "已有足够信息，确认结束", "tool": null, "arguments": {}}',
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("调研")

    assert result.completed is True
    assert "已有足够信息" in result.answer
    # 不应走"工具名为空"错误
    assert "工具名为空" not in result.answer


@pytest.mark.asyncio
async def test_final_confirm_gets_real_answer(registry):
    """
    收尾确认轮：模型第一次以 JSON 声明收尾（thought + tool:null）→
    被要求输出纯文本；第二次输出真实报告 → 最终回答是报告正文而非声明。
    """
    llm = ScriptedLLM([
        '{"thought": "信息足够，我将直接输出报告", "tool": null, "arguments": {}}',
        "## 摘要\n本报告基于实际阅读内容……",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("调研")

    assert result.completed is True
    # 最终回答是第二轮的纯文本报告，不是第一轮的收尾声明
    assert result.answer.startswith("## 摘要")
    assert "我将直接输出报告" not in result.answer
    # 确认提示确实写入过历史（要求纯文本）
    hints = [m.content for m in llm.llm_calls[1]
             if m.role == "user" and m.content.startswith("[任务提示]")]
    assert "纯文本" in hints[0]


@pytest.mark.asyncio
async def test_empty_tool_no_thought_rejected(registry):
    """空工具名且无 thought（完全退化）→ 报错并计入失败计数"""
    llm = ScriptedLLM([
        '{"tool": "", "arguments": {}}',  # 空 tool，无 thought
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


class FakeReader(BaseTool):
    """mock 的文件阅读工具（read_pdf）"""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_pdf",
            description="读取 PDF",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def run(self, path: str) -> str:
        return f"[PDF内容] {path} 的摘要"


def _make_tmp_files(tmp, names: list[str]):
    """在临时目录中创建空文件，并注册 list_files 工具"""
    import os
    from app.tools.builtin.codebase import ListFilesTool

    for name in names:
        with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
            f.write("dummy")
    return ListFilesTool(tmp)


@pytest.mark.asyncio
async def test_premature_final_questioned_once(registry):
    """
    提前收尾质疑：模型只读了部分文件就想收尾（tool=null+thought）→
    收到任务提示继续循环；再次收尾 → 接受（仅质疑一次，不强制、不死循环）。
    """
    import tempfile
    registry.register(_make_tmp_files(tempfile.mkdtemp(), ["a.pdf", "b.pdf", "c.pdf"]))
    registry.register(FakeReader())

    llm = ScriptedLLM([
        '{"thought": "读第一篇", "tool": "read_pdf", "arguments": {"path": "a.pdf"}}',
        '{"thought": "已完成分析", "tool": null, "arguments": {}}',
        '{"thought": "已补充阅读并完成", "tool": null, "arguments": {}}',
        '{"thought": "确认收尾", "tool": null, "arguments": {}}',
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("分析所有文章")

    # 收尾 → 软质疑 → 确认轮 → 再次收尾被接受
    assert result.completed is True
    assert result.steps == 4
    assert "确认收尾" in result.answer
    # 质疑提示写回消息历史，且只报事实（已读/总数）
    hints = [m.content for m in llm.llm_calls[3]
             if m.role == "user" and m.content.startswith("[任务提示]")]
    assert "1/3" in hints[0]
    # 确认轮要求纯文本回答
    assert "纯文本" in hints[1]


@pytest.mark.asyncio
async def test_no_question_when_all_read(registry):
    """已读文件数 == 清单数 → 收尾不质疑，直接接受"""
    import tempfile
    registry.register(_make_tmp_files(tempfile.mkdtemp(), ["a.pdf"]))
    registry.register(FakeReader())

    llm = ScriptedLLM([
        '{"thought": "读完了", "tool": "read_pdf", "arguments": {"path": "a.pdf"}}',
        '{"thought": "已完成分析", "tool": null, "arguments": {}}',
        '{"thought": "确认结束", "tool": null, "arguments": {}}',
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("分析")

    # 全部读完 → 无"未读完"质疑；收尾声明被确认轮拦截一次 → 第 3 次收尾被接受
    assert result.completed is True
    assert result.steps == 3
    assert len(llm.llm_calls) == 3
    # 只有确认轮提示（要求纯文本），没有"你已阅读 X/Y"的质疑提示
    assert not any("你已阅读" in m.content
                   for m in llm.llm_calls[1] if m.role == "user")


@pytest.mark.asyncio
async def test_compact_event_only_when_actually_folded(registry):
    """
    should_compact 触发但轮次不足（compact 返回原消息）→ 不记录 compact 事件。
    避免"未折叠却说已折叠"的观测误导。
    """
    from app.observability.tracer import Tracer
    from app.agent.context import SummaryCompressor

    llm = ScriptedLLM([
        '{"thought": "搜索", "tool": "search_kb", "arguments": {"query": "数据库"}}',
        "回答完毕。",
    ])
    agent = Agent(
        llm=llm, tools=registry, max_steps=5, tracer=Tracer(),
        compressor=SummaryCompressor(llm=llm, trigger_chars=50, keep_recent_rounds=1),
    )
    await agent.run("调研")

    # 仅 1 轮 → compact 返回原消息 → 不应记录事件
    events = [e.event for e in agent.tracer.events]
    assert "compact" not in events


@pytest.mark.asyncio
async def test_compact_event_when_folded(registry):
    """轮次足够且超阈值 → 实际折叠 → 记录 compact 事件 + 历史出现摘要"""
    from app.observability.tracer import Tracer
    from app.agent.context import SummaryCompressor

    llm = ScriptedLLM([
        '{"thought": "搜索1", "tool": "search_kb", "arguments": {"query": "数据库"}}',
        '{"thought": "搜索2", "tool": "search_kb", "arguments": {"query": "向量"}}',
        '{"thought": "搜索3", "tool": "search_kb", "arguments": {"query": "检索"}}',
        "回答完毕。",
    ])
    agent = Agent(
        llm=llm, tools=registry, max_steps=5, tracer=Tracer(),
        compressor=SummaryCompressor(llm=llm, trigger_chars=50, keep_recent_rounds=1),
    )
    await agent.run("调研")

    events = [e.event for e in agent.tracer.events]
    assert "compact" in events
    # 折叠后的历史含摘要（第 4 次循环调用时已压缩）
    history = llm.llm_calls[3]
    assert any("研究进展摘要" in m.content for m in history if m.role == "system")


@pytest.mark.asyncio
async def test_zero_read_final_blocked_until_read(registry):
    """
    0 阅读收尾 = 硬错误（无退路）：提示阅读，不放行"信息已足够"退路；
    模型读 1 篇后收尾走部分阅读质疑（一次），再次收尾被接受。
    """
    import tempfile
    registry.register(_make_tmp_files(tempfile.mkdtemp(), ["a.pdf", "b.pdf", "c.pdf"]))
    registry.register(FakeReader())

    llm = ScriptedLLM([
        '{"thought": "不想读了", "tool": null, "arguments": {}}',           # 0 读 → 硬拦
        '{"thought": "读一篇", "tool": "read_pdf", "arguments": {"path": "a.pdf"}}',
        '{"thought": "已读部分可以收尾", "tool": null, "arguments": {}}',   # 1/3 → 部分质疑
        '{"thought": "确认收尾", "tool": null, "arguments": {}}',           # 确认轮 → 要求纯文本
        '{"thought": "最终确认结束", "tool": null, "arguments": {}}',       # 接受
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("分析所有文章")

    assert result.completed is True
    assert result.steps == 5
    # ScriptedLLM 保存消息引用（llm_calls 均为最终态），按顺序收集全部提示断言
    hints = [m.content for m in llm.llm_calls[4]
             if m.role == "user" and m.content.startswith("[任务提示]")]
    assert len(hints) == 3
    # 硬拦提示在前：无"信息已足够"退路
    assert "尚未阅读任何文件" in hints[0]
    assert "若信息已足够" not in hints[0]
    # 部分阅读质疑：有退路且报已读/总数
    assert "1/3" in hints[1]
    # 收尾确认轮：要求纯文本回答
    assert "纯文本" in hints[2]


@pytest.mark.asyncio
async def test_zero_read_final_gives_up(registry):
    """持续 0 阅读收尾 → 失败计数用尽 → give_up（不无限循环、不放行编造）"""
    import tempfile
    registry.register(_make_tmp_files(tempfile.mkdtemp(), ["a.pdf", "b.pdf"]))
    registry.register(FakeReader())

    llm = ScriptedLLM([
        '{"thought": "不想读了", "tool": null, "arguments": {}}',
        '{"thought": "还是不想读", "tool": null, "arguments": {}}',
        '{"thought": "真的不读了", "tool": null, "arguments": {}}',
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5, max_parse_failures=3)
    result = await agent.run("分析所有文章")

    assert result.completed is False
    assert result.reason == "tool_errors"
    assert "未阅读" in result.answer


@pytest.mark.asyncio
async def test_list_files_duplicate_guidance(registry):
    """list_files 重复调用 → 提示文件列表已在历史中，引导直接阅读（防空转）"""
    import tempfile
    registry.register(_make_tmp_files(tempfile.mkdtemp(), ["a.pdf"]))

    llm = ScriptedLLM([
        '{"thought": "第一次列文件", "tool": "list_files", "arguments": {"pattern": "*"}}',
        '{"thought": "再列一次", "tool": "list_files", "arguments": {"pattern": "*"}}',
        "回答完毕。",
    ])
    agent = Agent(llm=llm, tools=registry, max_steps=5)
    result = await agent.run("调研")

    assert result.completed is True
    # 第二次 list_files（thought 不同，走工具级重复检测）被拦，提示引导直接阅读
    history = llm.llm_calls[2]
    hint = next(m for m in history if m.role == "user" and m.content.startswith("[工具错误"))
    assert "文件列表已存在于消息历史" in hint.content


@pytest.mark.asyncio
async def test_read_count_survives_compaction(registry):
    """
    压缩后已读文件数仍准确：折叠轮次的已读文件记入摘要清单，
    收尾质疑不会误报（"已读 5/5" 而非 "已读 1/5"）。
    """
    import tempfile, os
    from app.agent.context import SummaryCompressor

    tmp = tempfile.mkdtemp()
    for name in ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"]:
        with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
            f.write("dummy")
    registry.register(_make_tmp_files(tmp, ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"]))
    registry.register(FakeReader())

    llm = ScriptedLLM([
        *[
            f'{{"thought": "读{i}", "tool": "read_pdf", "arguments": {{"path": "{n}.pdf"}}}}'
            for i, n in enumerate(["a", "b", "c", "d", "e"])
        ],
        '{"thought": "全部读完了", "tool": null, "arguments": {}}',
        '{"thought": "确认结束", "tool": null, "arguments": {}}',
    ])
    agent = Agent(
        llm=llm, tools=registry, max_steps=10,
        compressor=SummaryCompressor(llm=llm, trigger_chars=50, keep_recent_rounds=1),
    )
    result = await agent.run("分析所有文章")

    # 5 篇全读 + 确认轮后收尾被接受
    assert result.completed is True
    assert result.steps == 7
    # 已读 5/5（清单代码级继承）→ 收尾不被"未读完"质疑；仅确认轮提示
    assert not any("你已阅读" in m.content
                   for m in llm.llm_calls[5] if m.role == "user")
