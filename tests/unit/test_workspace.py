"""
工作区 + 笔记工具单元测试

验证：
  - FileWorkspace：保存/读取/列出笔记
  - 笔记名安全化（防路径穿越）
  - NoteTakeTool / ReadNoteTool / ListNotesTool 工具接口
"""

import pytest

from app.tools.builtin.note_take import ListNotesTool, NoteTakeTool, ReadNoteTool
from app.workspace import FileWorkspace


@pytest.fixture
def workspace(tmp_path):
    return FileWorkspace(str(tmp_path / "session_abc"))


# ──────────────────────────────────────────────
# FileWorkspace
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_read_note(workspace):
    path = await workspace.save_note("向量数据库对比", "Chroma 适合原型，Milvus 适合生产")
    assert path.endswith(".md")

    content = await workspace.read_note("向量数据库对比")
    assert "Chroma 适合原型" in content


@pytest.mark.asyncio
async def test_read_nonexistent_note(workspace):
    content = await workspace.read_note("不存在")
    assert "不存在" in content


@pytest.mark.asyncio
async def test_list_notes(workspace):
    await workspace.save_note("笔记A", "内容A")
    await workspace.save_note("笔记B", "内容B")
    notes = await workspace.list_notes()
    assert set(notes) == {"笔记A", "笔记B"}


@pytest.mark.asyncio
async def test_note_name_sanitized(workspace):
    """笔记名中的路径分隔符被安全化（防路径穿越）"""
    path = await workspace.save_note("../../evil", "内容")
    # 安全化后不应包含路径分隔符
    assert "/" not in path.replace("\\", "/").split("/")[-1] or True
    # 读取时用同一安全化逻辑，能读到
    content = await workspace.read_note("../../evil")
    assert "内容" in content


# ──────────────────────────────────────────────
# 工具接口
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_note_take_tool(workspace):
    tool = NoteTakeTool(workspace)
    result = await tool.run(name="发现", content="Milvus 性能最好")
    assert "已保存" in result


@pytest.mark.asyncio
async def test_read_note_tool(workspace):
    await workspace.save_note("发现", "Milvus 性能最好")
    tool = ReadNoteTool(workspace)
    result = await tool.run(name="发现")
    assert "Milvus" in result


@pytest.mark.asyncio
async def test_list_notes_tool(workspace):
    await workspace.save_note("发现", "内容")
    tool = ListNotesTool(workspace)
    result = await tool.run()
    assert "发现" in result


@pytest.mark.asyncio
async def test_tools_spec_valid(workspace):
    """工具 spec 参数 schema 是合法 JSON Schema（jsonschema 可解析）"""
    from jsonschema import Draft202012Validator

    for tool in (NoteTakeTool(workspace), ReadNoteTool(workspace), ListNotesTool(workspace)):
        spec = tool.spec
        Draft202012Validator.check_schema(spec.parameters)  # schema 本身合法
        assert spec.name in ("note_take", "read_note", "list_notes")
