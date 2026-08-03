"""
LLM 客户端单元测试

mock 内部方法 _request，验证 chat/chat_with_response 的解析逻辑。
"""

from unittest.mock import AsyncMock

import pytest

from app.infrastructure.llm_client import OllamaChatClient, Message


@pytest.mark.asyncio
async def test_chat_returns_content():
    """
    正常情况：_request 返回合法 ChatResponse → chat() 返回 content 文本。
    """
    client = OllamaChatClient("http://test:11434", "test-model")
    client._request = AsyncMock(return_value=MockChatResponse("你好，我是助手"))

    result = await client.chat([
        Message(role="system", content="你是助手"),
        Message(role="user", content="你好"),
    ])

    assert result == "你好，我是助手"


@pytest.mark.asyncio
async def test_chat_with_multiple_messages():
    """
    多轮对话：验证消息被正确传递，并返回预期结果。
    """
    client = OllamaChatClient("http://test:11434", "test-model")
    client._request = AsyncMock(return_value=MockChatResponse("这是第二轮的回复"))

    messages = [
        Message(role="system", content="你是助手"),
        Message(role="user", content="第一轮问"),
        Message(role="assistant", content="第一轮答"),
        Message(role="user", content="第二轮问"),
    ]
    result = await client.chat(messages)

    assert result == "这是第二轮的回复"


@pytest.mark.asyncio
async def test_chat_handles_empty_response():
    """
    边界情况：Ollama 返回空消息 → 应返回空字符串。
    """
    client = OllamaChatClient("http://test:11434", "test-model")
    client._request = AsyncMock(return_value=MockChatResponse(""))

    result = await client.chat([Message(role="user", content="你好")])

    assert result == ""


# ──────────────────────────────────────────────
# 思考内容提取（think 字段 + 标签 fallback）
# ──────────────────────────────────────────────

def test_extract_think_tags_structured_field():
    """
    通道 1：message.thinking 结构化字段。
    llm_client 应将 thinking 与 content 分开返回。
    """
    from app.infrastructure.llm_client import ChatResponse

    # 模拟 Ollama 结构化响应（_request 层面已经解析好）
    resp = ChatResponse(
        content="最终回答。",
        thinking="让我分析一下这个问题的多个方面...",
    )
    assert resp.content == "最终回答。"
    assert resp.thinking == "让我分析一下这个问题的多个方面..."


def test_extract_think_tags_fallback_from_content():
    """
    通道 2：旧版 Ollama 无 thinking 字段，思考混在 content 的 <think> 标签里。
    _extract_think_tags 应提取标签内容并剥离标签。
    """
    from app.infrastructure.llm_client import _extract_think_tags

    text = "<think>我需要先检索知识库</think>最终回答。"
    thinking, content = _extract_think_tags(text)

    assert thinking == "我需要先检索知识库"
    assert content == "最终回答。"


def test_extract_think_tags_no_tags():
    """无 <think> 标签时原样返回"""
    from app.infrastructure.llm_client import _extract_think_tags

    thinking, content = _extract_think_tags("普通回答")
    assert thinking == ""
    assert content == "普通回答"


def test_extract_think_tags_multiline():
    """多行 think 标签应完整提取"""
    from app.infrastructure.llm_client import _extract_think_tags

    text = "<think>第一行思考\n第二行思考</think>正文"
    thinking, content = _extract_think_tags(text)
    assert "第一行思考" in thinking
    assert "第二行思考" in thinking
    assert content == "正文"


# 测试辅助：创建一个 ChatResponse 样式的对象
from app.infrastructure.llm_client import ChatResponse


def MockChatResponse(content: str) -> ChatResponse:
    return ChatResponse(content=content, done=True)
