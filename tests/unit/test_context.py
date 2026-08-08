"""
上下文管理单元测试

验证：
  1. TrimCompressor 的裁剪逻辑（短文本不动、超长截断、来源保留）
  2. orchestrator 集成：工具结果进入消息历史前被压缩
"""

import pytest

from app.agent.context import TrimCompressor
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


def test_long_result_truncated():
    """超长结果截断，保留开头（来源标记在前）"""
    compressor = TrimCompressor(max_chars=50)
    result = "[1] (path/to/doc.md)\n" + "内容" * 100
    compressed = compressor.compress_tool_result(result)

    assert len(compressed) <= 50 + len("\n...(结果过长已截断，可重新检索获取细节)")
    # 来源标记保留（可追溯、可重查）
    assert "[1] (path/to/doc.md)" in compressed
    # 截断标记存在
    assert "已截断" in compressed


def test_source_path_preserved():
    """来源路径（格式 [N] (path)）在截断后必须保留"""
    compressor = TrimCompressor(max_chars=30)
    result = "[1] (docs/arch.md)\n架构设计文档内容很长很长很长很长"
    compressed = compressor.compress_tool_result(result)
    assert "[1] (docs/arch.md)" in compressed


def test_should_compact_preview():
    """消息历史压缩预留接口：当前不触发"""
    compressor = TrimCompressor()
    assert compressor.should_compact([]) is False
    assert compressor.compact([]) == []


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
    """先调用工具，再回答的 mock LLM"""

    def __init__(self):
        self.calls = []

    async def chat_with_response(self, messages):
        from app.infrastructure.llm_client import ChatResponse
        self.calls.append(messages)
        if len(self.calls) == 1:
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

    # 第二次 LLM 调用时，消息历史里的工具结果应是压缩后的
    history = llm.calls[1]
    tool_msg = next(m for m in history if m.role == "user" and m.content.startswith("[工具结果"))
    assert "已截断" in tool_msg.content
    assert len(tool_msg.content) < 300  # 原始结果 1500+ 字符 → 压缩到 ~100
    # 来源保留
    assert "[1] (docs/arch.md)" in tool_msg.content
