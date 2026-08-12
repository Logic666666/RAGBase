"""
代码库路径解析公共工具测试

关键回归：完整路径（search_kb 返回格式 ./data/kb/xxx/source/src/main.py）
必须能解析到真实文件——曾因拼接路径未校验存在性导致全部解析失败。
"""

import os

import pytest

from app.tools.path_utils import resolve_codebase_path


@pytest.fixture
def codebase(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    return str(tmp_path)


def test_relative_path(codebase):
    """相对路径正常解析"""
    result = resolve_codebase_path(codebase, "src/main.py")
    assert result is not None
    assert os.path.isfile(result)


def test_full_path(codebase):
    """
    完整路径（./data/kb/xxx/source/src/main.py 格式）正常解析——
    回归：曾因拼接路径未校验存在性，完整路径永远解析失败。
    """
    full = os.path.join(codebase, "src", "main.py")
    result = resolve_codebase_path(codebase, full)
    assert result is not None
    assert os.path.isfile(result)
    # 与相对路径解析到同一文件
    assert os.path.realpath(result) == os.path.realpath(full)


def test_nonexistent_returns_none(codebase):
    """不存在的文件返回 None（不返回拼接出的假路径）"""
    assert resolve_codebase_path(codebase, "src/no_such.py") is None


def test_traversal_blocked(codebase):
    """路径穿越拒绝"""
    assert resolve_codebase_path(codebase, "../../etc/passwd") is None


def test_outside_codebase_returns_none(codebase, tmp_path):
    """代码库外的完整路径拒绝（文件放在 codebase 目录之外）"""
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    assert resolve_codebase_path(codebase, str(outside)) is None
