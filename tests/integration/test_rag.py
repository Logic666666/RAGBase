"""
RAG 管线集成测试

验证 RagService 的编排逻辑：检索 → 构建上下文 → 调 LLM → 整理来源。
嵌入和 LLM 都 mock，只测"胶水代码"是否正确。
"""

from unittest.mock import AsyncMock

import pytest

from app.infrastructure.llm_client import ChatResponse
from app.services.rag import RagService


@pytest.fixture
def rag_with_mocks(settings):
    """
    创建 RagService，embedding 和 LLM 都用 mock。

    - mock embedding：避免真实调用 Ollama
    - mock llm._request：避免真实调用 Ollama
    """
    rag = RagService(settings)

    # Mock embedding
    mock_embed = AsyncMock()
    mock_embed.embed_documents.side_effect = (
        lambda texts: [[0.1 + i * 0.1] * 10 for i in range(len(texts))]
    )
    mock_embed.embed_query.return_value = [0.1] * 10
    rag.vs.embedding_client = mock_embed

    # Mock LLM
    rag.llm._request = AsyncMock(
        return_value=ChatResponse(content="Python是一种编程语言。", done=True)
    )

    return rag


@pytest.mark.asyncio
async def test_answer_question_returns_correct_format(rag_with_mocks):
    """验证返回格式：answer (str) + sources (list[dict])"""
    persist_dir = _kb_vector_dir(rag_with_mocks.settings, "test_kb")
    await rag_with_mocks.vs.add_documents(persist_dir, [
        ("Python是一种编程语言", {"source": "python_intro.txt"}),
    ])

    answer, sources = await rag_with_mocks.answer_question(
        "test_kb", "什么是Python？", top_k=2,
    )

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert isinstance(sources, list)
    assert len(sources) > 0
    for src in sources:
        assert "source" in src
        assert "snippet" in src


@pytest.mark.asyncio
async def test_answer_question_returns_sources(rag_with_mocks):
    """验证来源信息正确传递"""
    persist_dir = _kb_vector_dir(rag_with_mocks.settings, "test_kb")
    await rag_with_mocks.vs.add_documents(persist_dir, [
        ("Python由Guido van Rossum创建。", {"source": "history.txt"}),
    ])

    answer, sources = await rag_with_mocks.answer_question(
        "test_kb", "谁创造了Python？", top_k=1,
    )

    source_files = [s["source"] for s in sources]
    assert any("history.txt" in s for s in source_files)


def _kb_vector_dir(settings, kb_name: str) -> str:
    """模拟 RagService._kb_vector_dir 的路径计算"""
    import os
    return os.path.join(settings.data_dir, "vectorstore", kb_name)
