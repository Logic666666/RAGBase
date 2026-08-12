"""
KnowledgeBaseTool 自我拒绝机制测试

验证：search_kb 连续命中代码文件达到阈值后"自我拒绝"，
迫使模型查阅工具列表换工具（不点名其他工具）。
"""

import pytest

from app.tools.builtin.knowledge_base import KnowledgeBaseTool


class FakeRag:
    """mock RagService，可控制返回代码文件或文档"""

    def __init__(self, sources):
        self.sources = sources

    async def search_docs(self, kb, query, top_k=4):
        return [{"text": f"内容{i}", "source": s, "score": 0.9} for i, s in enumerate(self.sources)]


@pytest.mark.asyncio
async def test_first_code_hit_returns_hint():
    """第一次命中代码文件：返回提示但不拒绝"""
    rag = FakeRag(["./data/kb/x/source/src/main.py"])
    tool = KnowledgeBaseTool(rag, "x")

    result = await tool.run("查询")
    assert "代码文件" in result
    assert "停止使用" not in result  # 未拒绝


@pytest.mark.asyncio
async def test_second_code_hit_rejects():
    """连续第二次命中代码文件：自我拒绝（提示停止使用本工具）"""
    rag = FakeRag(["./data/kb/x/source/src/main.py"])
    tool = KnowledgeBaseTool(rag, "x")

    await tool.run("查询1")
    result = await tool.run("查询2")
    assert "停止使用本工具" in result
    assert "查阅可用工具列表" in result
    # 不点名其他具体工具
    assert "grep_code" not in result and "read_file" not in result


@pytest.mark.asyncio
async def test_doc_hit_resets_counter():
    """命中文档后重置计数：工具恢复可用"""
    rag = FakeRag(["./data/kb/x/source/src/main.py"])
    tool = KnowledgeBaseTool(rag, "x")

    await tool.run("查询1")  # 代码文件
    await tool.run("查询2")  # 代码文件 → 拒绝

    # 换一个返回文档的 rag
    rag2 = FakeRag(["./data/kb/x/source/README.md"])
    tool.rag = rag2
    result = await tool.run("查询3")
    assert "停止使用" not in result  # 未拒绝，计数已重置

    # 再命中代码文件，从 0 重新计数
    rag3 = FakeRag(["./data/kb/x/source/src/main.py"])
    tool.rag = rag3
    r1 = await tool.run("查询4")
    assert "停止使用" not in r1  # 第一次，仅提示
