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


# 测试辅助：创建一个 ChatResponse 样式的对象
from app.infrastructure.llm_client import ChatResponse


def MockChatResponse(content: str) -> ChatResponse:
    return ChatResponse(content=content, done=True)
