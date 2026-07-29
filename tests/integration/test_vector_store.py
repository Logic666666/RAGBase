"""
向量存储集成测试

用真实的 ChromaDB，但 mock embedding 层。
"""

import os
from unittest.mock import AsyncMock

import pytest

from app.infrastructure.vector_store import VectorStore


@pytest.fixture
def vs(settings):
    """
    创建一个 VectorStore，用 mock 的 embedding_client。

    embed_documents 根据输入文本数量动态生成对应数量的向量，
    而不是固定返回 3 个。
    """
    store = VectorStore(settings)
    mock_embed = AsyncMock()
    mock_embed.embed_documents.side_effect = (
        lambda texts: [[0.1 + i * 0.1] * 10 for i in range(len(texts))]
    )
    mock_embed.embed_query.return_value = [0.1] * 10
    store.embedding_client = mock_embed
    return store


@pytest.mark.asyncio
async def test_add_and_search(vs, tmp_path):
    """核心流程：添加 3 条 → 搜索 top_k=2 → 返回 ≤2 条"""
    persist_dir = os.path.join(tmp_path, "chroma_test")
    await vs.add_documents(persist_dir, [
        ("今天天气真好", {"source": "test1.txt"}),
        ("明天要下雨了", {"source": "test2.txt"}),
        ("Python是一种编程语言", {"source": "test3.txt"}),
    ])
    results = await vs.similarity_search(persist_dir, "天气", top_k=2)
    assert len(results) <= 2
    assert len(results) > 0
    for text, meta, dist in results:
        assert isinstance(text, str)
        assert isinstance(meta, dict)
        assert isinstance(dist, float)


@pytest.mark.asyncio
async def test_search_empty_db_returns_empty(vs, tmp_path):
    """空数据库搜索应返回空列表"""
    results = await vs.similarity_search(os.path.join(tmp_path, "chroma_empty"), "anything", top_k=4)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_multiple_adds_accumulate(vs, tmp_path):
    """多次添加：文档应累加"""
    persist_dir = os.path.join(tmp_path, "chroma_multi")
    await vs.add_documents(persist_dir, [("第一组文档A", {"source": "a.txt"})])
    await vs.add_documents(persist_dir, [
        ("第二组文档B", {"source": "b.txt"}),
        ("第二组文档C", {"source": "c.txt"}),
    ])
    results = await vs.similarity_search(persist_dir, "文档", top_k=10)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_delete_collection(vs, tmp_path):
    """删除集合后搜索应返回空"""
    persist_dir = os.path.join(tmp_path, "chroma_delete")
    await vs.add_documents(persist_dir, [("待删除的文档", {"source": "delete_me.txt"})])
    before = await vs.similarity_search(persist_dir, "待删除", top_k=5)
    assert len(before) == 1
    vs.delete_collection(persist_dir)
    after = await vs.similarity_search(persist_dir, "待删除", top_k=5)
    assert len(after) == 0
