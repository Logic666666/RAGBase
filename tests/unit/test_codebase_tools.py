"""
代码库工具单元测试

验证：
  - grep_code：正则搜索、无匹配提示、限制结果数
  - read_file：带行号读取、路径穿越防护
  - list_files：通配符匹配
"""

import os

import pytest

from app.tools.builtin.codebase import GrepCodeTool, ReadFileTool, ListFilesTool


@pytest.fixture
def codebase(tmp_path):
    """构造一个临时代码库"""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "import sys\n\ndef main():\n    print('hello')\n", encoding="utf-8"
    )
    (src / "utils.py").write_text(
        "def process_frame(frame):\n    return frame\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "# 项目说明\n这是一个测试项目\n", encoding="utf-8"
    )
    return str(tmp_path)


# ──────────────────────────────────────────────
# grep_code
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_grep_finds_matches(codebase):
    tool = GrepCodeTool(codebase)
    result = await tool.run("def ")
    assert "main.py:3: def main()" in result
    assert "utils.py:1: def process_frame" in result


@pytest.mark.asyncio
async def test_grep_no_match(codebase):
    tool = GrepCodeTool(codebase)
    result = await tool.run("不存在的符号xyz")
    assert "未找到匹配" in result


@pytest.mark.asyncio
async def test_grep_no_match_shows_file_structure(codebase):
    """
    grep 无结果时返回按目录分组的文件结构，
    引导模型按文件名选择文件直接阅读（不依赖模型自觉）。
    """
    tool = GrepCodeTool(codebase)
    result = await tool.run("不存在的符号xyz")
    # 包含文件结构提示和文件名
    assert "项目文件结构" in result
    assert "src/" in result
    assert "main.py" in result
    # 明确引导 read_file
    assert "read_file" in result


@pytest.mark.asyncio
async def test_grep_limits_results(codebase):
    tool = GrepCodeTool(codebase)
    result = await tool.run("def |print|import", max_results=1)
    # 只返回 1 行
    assert result.count("\n") == 0 or len(result.splitlines()) == 1


@pytest.mark.asyncio
async def test_grep_invalid_regex(codebase):
    tool = GrepCodeTool(codebase)
    result = await tool.run("([")
    assert "正则表达式无效" in result


# ──────────────────────────────────────────────
# read_file
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_file_with_line_numbers(codebase):
    tool = ReadFileTool(codebase)
    result = await tool.run("src/main.py")
    assert "1 | import sys" in result
    assert "3 | def main():" in result


@pytest.mark.asyncio
async def test_read_file_path_traversal_blocked(codebase):
    """路径穿越防护：../../ 不能读到代码库外的文件"""
    tool = ReadFileTool(codebase)
    # 尝试读取代码库外的文件
    result = await tool.run("../../secret.txt")
    assert "非法路径" in result


@pytest.mark.asyncio
async def test_read_file_accepts_full_path(codebase):
    """
    完整路径容错：search_kb 返回的 metadata.source 格式
    （./data/kb/xxx/source/src/main.py）也应能读取。
    """
    tool = ReadFileTool(codebase)
    full_path = os.path.join(codebase, "src", "main.py")
    result = await tool.run(full_path)
    assert "def main():" in result
    # 绝对路径也行
    abs_result = await tool.run(os.path.abspath(full_path))
    assert "def main():" in abs_result


@pytest.mark.asyncio
async def test_read_file_nonexistent(codebase):
    tool = ReadFileTool(codebase)
    result = await tool.run("no_such_file.py")
    assert "文件不存在" in result


@pytest.mark.asyncio
async def test_read_file_truncates_long(codebase):
    tool = ReadFileTool(codebase, )
    # 用 max_chars 限制
    long_file = os.path.join(codebase, "long.txt")
    with open(long_file, "w", encoding="utf-8") as f:
        f.write("x" * 5000)
    result = await tool.run("long.txt", max_chars=100)
    assert "已截断" in result


# ──────────────────────────────────────────────
# list_files
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_files_all(codebase):
    tool = ListFilesTool(codebase)
    result = await tool.run()
    assert "src/main.py" in result
    assert "README.md" in result


@pytest.mark.asyncio
async def test_list_files_pattern(codebase):
    tool = ListFilesTool(codebase)
    result = await tool.run("*.py")
    assert "src/main.py" in result
    assert "README.md" not in result  # 非 py 文件被过滤


@pytest.mark.asyncio
async def test_list_files_no_match(codebase):
    tool = ListFilesTool(codebase)
    result = await tool.run("*.java")
    assert "没有匹配" in result
