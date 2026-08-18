"""
上下文管理单元测试

验证：
  1. TrimCompressor 的裁剪逻辑（短文本不动、超长截断、来源保留）
  2. orchestrator 集成：工具结果进入消息历史前被压缩
"""

import pytest

from app.agent.context import SummaryCompressor, TrimCompressor
from app.agent.orchestrator import Agent
from app.tools.base import BaseTool, ToolSpec
from app.tools.registry import ToolRegistry


# ──────────────────────────────────────────────
# TrimCompressor
# ──────────────────────────────────────────────

def test_short_result_unchanged():
    """短结果不裁剪"""
    compressor = TrimCompressor(max_chars=100)
    result = "[1] (path/to/doc.md)\n简短内容"
    assert compressor.compress_tool_result(result) == result


def test_long_result_truncated_with_total():
    """超长结果截断，保留开头且提示必须包含总量（模型判断是否需要重取）"""
    compressor = TrimCompressor(max_chars=50)
    result = "[1] (path/to/doc.md)\n" + "内容" * 100
    compressed = compressor.compress_tool_result(result)

    assert len(compressed) <= 50 + 100  # 截断 + 提示语
    # 来源标记保留（可追溯、可重查）
    assert "[1] (path/to/doc.md)" in compressed
    # 截断提示包含总量（模型据此判断完整性）
    assert "已截断" in compressed
    assert "共 " in compressed
    assert str(len(result)) in compressed  # 总量数字


def test_source_path_preserved():
    """来源路径（格式 [N] (path)）在截断后必须保留"""
    compressor = TrimCompressor(max_chars=30)
    result = "[1] (docs/arch.md)\n架构设计文档内容很长很长很长很长"
    compressed = compressor.compress_tool_result(result)
    assert "[1] (docs/arch.md)" in compressed


def test_large_threshold_keeps_file_list():
    """
    兜底阈值足够大：文件列表（约 2KB）不应被截断——
    结构化信息（路径）截断会导致模型看不到完整项目结构。
    """
    compressor = TrimCompressor()  # 默认 8000
    file_list = "\n".join(f"src/module_{i}/file_{i}.py" for i in range(73))
    assert len(file_list) < 8000
    assert compressor.compress_tool_result(file_list) == file_list


def test_should_compact_preview():
    """消息历史压缩预留接口：当前不触发"""
    compressor = TrimCompressor()
    assert compressor.should_compact([]) is False
    import asyncio
    assert asyncio.run(compressor.compact([])) == []


# ──────────────────────────────────────────────
# SummaryCompressor（消息历史摘要）
# ──────────────────────────────────────────────

class FakeLLM:
    """返回固定摘要的 mock LLM（可记录输入、可模拟失败）"""

    def __init__(self, summary="已收集关键信息A、B，来源文件x.py",
                 raise_on_chat: bool = False):
        self.summary = summary
        self.calls = 0
        self.inputs = []          # 记录每次 chat 的消息列表（验证摘要继承）
        self.raise_on_chat = raise_on_chat

    async def chat(self, messages):
        self.calls += 1
        self.inputs.append(messages)
        if self.raise_on_chat:
            raise RuntimeError("摘要模型不可用")
        return self.summary


def _make_history(rounds: int, content_len: int = 8):
    """构造消息历史：system + user(任务) + N 轮 (assistant+user)"""
    from app.infrastructure.llm_client import Message
    messages = [
        Message(role="system", content="工具列表"),
        Message(role="user", content="调研任务"),
    ]
    for i in range(rounds):
        messages.append(Message(
            role="assistant",
            content=f'{{"tool": "search_kb", "arguments": {{"query": "q{i}"}}}}',
        ))
        messages.append(Message(
            role="user",
            content=f"[工具结果 search_kb]\n第{i}轮" + "内容" * content_len,
        ))
    return messages


def test_summary_should_compact_by_chars():
    """轮次字符总量超阈值才触发（按字符量估算，不是按轮数）"""
    llm = FakeLLM()
    compressor = SummaryCompressor(llm=llm, trigger_chars=1000, keep_recent_rounds=1)

    assert compressor.should_compact(_make_history(5, content_len=8)) is False   # 总量小
    assert compressor.should_compact(_make_history(5, content_len=200)) is True  # 总量超


@pytest.mark.asyncio
async def test_summary_compacts_early_rounds():
    """折叠早期轮次为摘要，保留最近 N 轮原文 + system 锚点"""
    llm = FakeLLM()
    compressor = SummaryCompressor(llm=llm, trigger_chars=200, keep_recent_rounds=1)

    history = _make_history(5, content_len=50)  # 每轮 ~170 字符，总量超 200
    result = await compressor.compact(history)

    roles = [m.role for m in result]
    # system 锚点保留 + user(任务) + 摘要 + 最近 1 轮(assistant+user)
    assert roles.count("system") == 2  # 工具列表 + 摘要
    assert "研究进展摘要" in result[2].content
    # 最近 1 轮原文保留（assistant + user 对）
    assert result[-2].role == "assistant"
    assert result[-1].content.startswith("[工具结果")
    # 早期轮次被折叠（消息数大幅减少）
    assert len(result) < len(history)
    # LLM 被调用来生成摘要
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_summary_keeps_all_system_anchors():
    """所有 system 锚点（工具列表/文件结构/研究计划）必须保留"""
    from app.infrastructure.llm_client import Message
    llm = FakeLLM()
    compressor = SummaryCompressor(llm=llm, trigger_chars=200, keep_recent_rounds=1)

    history = [
        Message(role="system", content="工具列表"),
        Message(role="system", content="项目文件结构"),
        Message(role="system", content="研究计划"),
        Message(role="user", content="任务"),
    ]
    for i in range(4):
        history.append(Message(role="assistant", content=f'{{"tool": "x"}}'))
        history.append(Message(role="user", content=f"[工具结果 x]\n结果{i}"))

    result = await compressor.compact(history)
    system_contents = [m.content for m in result if m.role == "system"]
    assert "工具列表" in system_contents[0]
    assert "项目文件结构" in system_contents[1]
    assert "研究计划" in system_contents[2]
    assert "研究进展摘要" in system_contents[3]


@pytest.mark.asyncio
async def test_no_immediate_recompress_after_compact():
    """压缩后残余仍超阈值也不立即再次压缩（增量节流防 thrashing）"""
    llm = FakeLLM()
    compressor = SummaryCompressor(llm=llm, trigger_chars=200, keep_recent_rounds=1)

    history = _make_history(5, content_len=50)  # 总量 850+ > 200
    assert compressor.should_compact(history) is True
    result = await compressor.compact(history)

    # 压缩后剩余 1 轮原文 + 摘要，可能仍超阈值，但新增轮次不足 → 不立即再压
    assert compressor.should_compact(result) is False


@pytest.mark.asyncio
async def test_old_summary_folded_into_new():
    """重复压缩：旧摘要并入折叠集合一起总结（摘要继承，只留一条摘要）"""
    llm = FakeLLM()
    compressor = SummaryCompressor(llm=llm, trigger_chars=200, keep_recent_rounds=1)

    history = _make_history(5, content_len=50)
    once = await compressor.compact(history)
    assert llm.calls == 1

    # 再新增 5 轮（增量 ≥ keep 轮）→ 再次触发
    extended = once + _make_history(5, content_len=50)[2:]
    assert compressor.should_compact(extended) is True

    result2 = await compressor.compact(extended)
    # 只留一条摘要（旧摘要被合并进新摘要，不累积）
    summaries = [m for m in result2
                 if m.role == "system" and m.content.startswith("研究进展摘要")]
    assert len(summaries) == 1
    # 旧摘要进入了摘要生成输入（FakeLLM 第二次调用能看到旧摘要文本）
    assert "已收集关键信息" in llm.inputs[-1][-1].content


@pytest.mark.asyncio
async def test_compact_failure_degrades():
    """摘要生成失败 → 返回原消息（降级不阻塞主任务），下次可重试"""
    llm = FakeLLM(raise_on_chat=True)
    compressor = SummaryCompressor(llm=llm, trigger_chars=200, keep_recent_rounds=1)

    history = _make_history(5, content_len=50)
    result = await compressor.compact(history)

    assert result == history          # 原样返回，不抛异常
    assert compressor.should_compact(history) is True  # 节流基线未更新，仍可重试


def _make_read_history(rounds: int, content_len: int = 20):
    """构造含 read_pdf 轮次的历史（assistant 消息携带文件路径）"""
    from app.infrastructure.llm_client import Message
    messages = [
        Message(role="system", content="工具列表"),
        Message(role="user", content="调研任务"),
    ]
    for i in range(rounds):
        messages.append(Message(role="assistant", content=(
            f'{{"tool": "read_pdf", "arguments": {{"path": "doc{i}.pdf"}}}}'
        )))
        messages.append(Message(role="user", content=(
            f"[工具结果 read_pdf]\ndoc{i}.pdf 内容" + "长" * content_len
        )))
    return messages


@pytest.mark.asyncio
async def test_summary_includes_read_files():
    """
    摘要消息附带已读文件清单（只列折叠轮次的）：
    压缩后模型据此知道"读过哪些文件"，不会重读被折叠的文档
    而与收敛检测（seen_calls 全程记忆）冲突。
    """
    llm = FakeLLM()
    compressor = SummaryCompressor(llm=llm, trigger_chars=200, keep_recent_rounds=1)

    history = _make_read_history(5)  # 折叠前 4 轮，保留最近 1 轮
    result = await compressor.compact(history)

    summary_msg = next(m for m in result if m.content.startswith("研究进展摘要"))
    assert "已阅读文件" in summary_msg.content
    assert "doc0.pdf" in summary_msg.content  # 折叠轮的文件在清单中
    assert "doc3.pdf" in summary_msg.content
    assert "doc4.pdf" not in summary_msg.content  # 保留轮原文可见，不列清单（避免重复计数）


@pytest.mark.asyncio
async def test_summary_skips_failed_reads():
    """
    失败/被拦截的 read 调用不计入已读清单：
    收敛检测保证成功调用必然是新文件，失败调用（如重复调用被拦）
    列入清单会造成"已阅读文件：A、A"的重复。
    """
    from app.infrastructure.llm_client import Message
    llm = FakeLLM()
    compressor = SummaryCompressor(llm=llm, trigger_chars=200, keep_recent_rounds=1)

    history = _make_read_history(7)  # doc0-6 全部"成功"
    # 第 1 轮改为失败（模拟重复调用被拦）：doc0 未读成，不应进清单
    history[3] = Message(role="user", content="[工具错误 read_pdf]\n重复调用")

    result = await compressor.compact(history)  # 折叠前 6 轮（含失败轮 doc0）

    summary_msg = next(m for m in result if m.content.startswith("研究进展摘要"))
    assert "已阅读文件：doc1.pdf" in summary_msg.content  # 成功轮进清单
    assert summary_msg.content.count("doc0.pdf") == 0     # 失败轮 doc0 不在清单
    # 清单不重复（每个成功文件只出现一次）
    for name in ["doc1.pdf", "doc2.pdf"]:
        assert summary_msg.content.count(name) == 1


@pytest.mark.asyncio
async def test_read_files_survive_llm_forgetting():
    """
    清单由代码维护，不依赖 LLM 复述：
    第二次压缩后清单 = 旧文件 + 新文件（即使摘要 LLM 未复述旧清单行）。
    """
    llm = FakeLLM()  # 固定摘要文本（不保留输入）——模拟模型复述失真
    compressor = SummaryCompressor(llm=llm, trigger_chars=200, keep_recent_rounds=1)

    history = _make_read_history(5, content_len=50)
    once = await compressor.compact(history)  # 折叠 doc0-3，清单：doc0-3
    extended = once + _make_read_history(5, content_len=50)[2:]
    result2 = await compressor.compact(extended)  # 折叠 + 继承

    summary_msg = next(m for m in result2 if m.content.startswith("研究进展摘要"))
    assert "doc0.pdf" in summary_msg.content   # 旧清单保留（LLM 没复述也不丢）
    assert "doc4.pdf" in summary_msg.content   # 新折叠加入
    assert summary_msg.content.count("doc0.pdf") == 1  # 全量清单不重复


# ──────────────────────────────────────────────
# orchestrator 集成
# ──────────────────────────────────────────────

class FakeKB(BaseTool):
    """返回超长结果的 mock 工具"""

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
        # 模拟超长检索结果（来源 + 超长正文）
        return "[1] (docs/arch.md)\n" + "架构内容" * 500


class ScriptedLLM:
    """先规划，再调用工具，再回答的 mock LLM"""

    def __init__(self):
        self.chat_calls = []   # 规划轮
        self.llm_calls = []    # ReAct 循环

    async def chat(self, messages):
        self.chat_calls.append(messages)
        return '{"plan": [{"dimension": "架构", "keywords": ["架构"]}]}'

    async def chat_with_response(self, messages):
        from app.infrastructure.llm_client import ChatResponse
        self.llm_calls.append(messages)
        if len(self.llm_calls) == 1:
            return ChatResponse(content='{"tool": "search_kb", "arguments": {"query": "架构"}}')
        return ChatResponse(content="回答完毕。")


@pytest.mark.asyncio
async def test_tool_result_compressed_in_history():
    """工具结果进入消息历史前必须被压缩"""
    registry = ToolRegistry()
    registry.register(FakeKB())

    # 压缩器限 100 字符
    llm = ScriptedLLM()
    agent = Agent(llm=llm, tools=registry, max_steps=5,
                  compressor=TrimCompressor(max_chars=100))
    await agent.run("分析项目架构")

    # 第二次循环调用时，消息历史里的工具结果应是压缩后的
    history = llm.llm_calls[1]
    tool_msg = next(m for m in history if m.role == "user" and m.content.startswith("[工具结果"))
    assert "已截断" in tool_msg.content
    assert len(tool_msg.content) < 300  # 原始结果 1500+ 字符 → 压缩到 ~100
    # 来源保留
    assert "[1] (docs/arch.md)" in tool_msg.content
