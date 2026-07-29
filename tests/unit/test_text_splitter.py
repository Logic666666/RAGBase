"""
文本分块器单元测试

测试策略：纯函数，不需要 mock，不需要外部依赖。
只管给什么输入、期望什么输出。

分层目的：
  unit/ 层的测试应该快速（毫秒级）、可靠（从不因为网络而失败）。
  text_splitter 是纯算法，是 unit 层最理想的测试对象。
"""

from app.infrastructure.text_splitter import split_text


# ──────────────────────────────────────────────
# 边界条件测试
# ──────────────────────────────────────────────

def test_empty_text():
    """空文本应返回空列表"""
    assert split_text("") == []


def test_short_text():
    """短于 chunk_size 的文本应保持为一个块"""
    result = split_text("hello world", chunk_size=100)
    assert result == ["hello world"]


def test_exact_chunk_size():
    """文本正好等于 chunk_size 时保持一个块"""
    text = "x" * 500
    result = split_text(text, chunk_size=500, chunk_overlap=0)
    assert result == [text]


# ──────────────────────────────────────────────
# 切分逻辑测试
# ──────────────────────────────────────────────

def test_long_text_splits_into_multiple_chunks():
    """长文本应按 chunk_size 切分成多个块"""
    result = split_text("A" * 2000, chunk_size=500, chunk_overlap=0)
    assert len(result) == 4
    assert all(len(c) <= 500 for c in result)


def test_chinese_period_splitting():
    """中文文本应按句号切分"""
    # "aaa。"=4 字符，chunk_size=6 时合并会跨块
    text = "aaa。bbb。ccc。ddd。"
    result = split_text(text, chunk_size=6, chunk_overlap=0)
    # 应该切分成多个块（每块 ~4 字符，合并到接近 6）
    assert len(result) >= 2, f"预期 >=2 块，实际 {len(result)}"


def test_paragraph_split_takes_priority():
    """段落分隔符（\\n\\n）优先级高于句号"""
    text = "第一段。\n\n第二段。\n\n第三段。"
    result = split_text(text, chunk_size=300, chunk_overlap=0)
    # 段落应保持完整，每块包含一个段落
    assert "第一段。" in result[0]


# ──────────────────────────────────────────────
# Overlap 测试
# ──────────────────────────────────────────────

def test_overlap_is_applied():
    """相邻块之间应包含重叠文本"""
    result = split_text(
        "A" * 200 + "B" * 200 + "C" * 200,
        chunk_size=300,
        chunk_overlap=50,
    )
    # 有 overlap 时，相邻块的内容会交叉
    assert len(result) >= 2
    # 验证前一块末尾和后一块开头有重叠
    has_both = any("A" in c and "B" in c for c in result)
    assert has_both, "overlap 应让相邻块内容重叠"


def test_no_overlap_when_set_to_zero():
    """chunk_overlap=0 时不应重叠"""
    result = split_text("A" * 200 + "B" * 200, chunk_size=250, chunk_overlap=0)
    # 各块应该清晰地分开
    assert len(result) >= 2


# ──────────────────────────────────────────────
# 兜底策略测试
# ──────────────────────────────────────────────

def test_hard_split_fallback():
    """空字符串分隔符（兜底）应按 chunk_size 硬切"""
    result = split_text("x" * 2500, separators=[""], chunk_size=500, chunk_overlap=0)
    assert len(result) == 5
    assert all(len(c) == 500 for c in result)


def test_separator_not_found_uses_next():
    """当前分隔符不存在时，自动降级使用下一级"""
    text = "hello world this is a test"
    # 用 "\n\n" 和 "\n" 都切不动（文本中没有），会降级到空格
    result = split_text(
        text,
        separators=["\n\n", "\n", " "],
        chunk_size=10,
        chunk_overlap=0,
    )
    # 应该在空格处切分
    assert len(result) >= 2
