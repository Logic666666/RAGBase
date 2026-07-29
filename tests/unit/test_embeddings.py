"""
嵌入客户端单元测试

测试策略：mock 内部方法 _request，不碰真实的 httpx。
这样测试验证的是"embed_query 是否能正确解析 _request 的返回值"，

不验证"httpx 能不能调对"——那是 httpx 库自己的测试。
"""

from unittest.mock import AsyncMock

import pytest

from app.infrastructure.embeddings import OllamaEmbeddingClient


@pytest.mark.asyncio
async def test_embed_query_returns_expected_vector():
    """
    正常情况：_request 返回合法数据 → embed_query 正确提取 embedding。
    """
    client = OllamaEmbeddingClient("http://test:11434", "test-model")
    client._request = AsyncMock(return_value={"embedding": [0.1, 0.2, 0.3]})

    result = await client.embed_query("hello")

    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_query_raises_on_missing_embedding():
    """
    异常情况：_request 返回缺少 'embedding' 字段的响应。
    代码应抛出 ValueError，而不是悄无声息地返回 None。
    """
    client = OllamaEmbeddingClient("http://test:11434", "test-model")
    client._request = AsyncMock(return_value={"prompt": "hello"})  # 缺 embedding 字段

    with pytest.raises(ValueError, match="缺少 'embedding' 字段"):
        await client.embed_query("hello")


@pytest.mark.asyncio
async def test_embed_documents_returns_list_of_vectors():
    """
    批量接口：embed_documents 应为每个输入文本返回一个向量。
    """
    client = OllamaEmbeddingClient("http://test:11434", "test-model")
    client._request = AsyncMock(return_value={"embedding": [0.1, 0.2, 0.3]})

    results = await client.embed_documents(["hello", "world"])

    assert len(results) == 2
    assert results[0] == [0.1, 0.2, 0.3]
    assert results[1] == [0.1, 0.2, 0.3]
