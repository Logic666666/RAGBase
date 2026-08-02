"""
工具调用解析器单元测试

验证 parse_tool_call() 的判定逻辑：
  - 合法工具调用 JSON → 解析为 ToolCall
  - 普通回答文本 → None
  - 格式错误的 JSON → 宽容处理
"""

from app.agent.schemas import parse_tool_call


def test_valid_tool_call():
    """合法工具调用应解析为 ToolCall"""
    result = parse_tool_call(
        '{"thought": "先搜索", "tool": "search_kb", "arguments": {"query": "数据库"}}'
    )
    assert result is not None
    assert result.tool == "search_kb"
    assert result.arguments == {"query": "数据库"}
    assert result.thought == "先搜索"


def test_plain_answer_returns_none():
    """普通回答（非 JSON）应返回 None"""
    assert parse_tool_call("根据检索结果，推荐使用 ChromaDB。") is None


def test_malformed_json_returns_none():
    """不完整的 JSON 应宽容处理为 None"""
    assert parse_tool_call('{"tool": "search_kb", "arguments"') is None


def test_markdown_wrapped_json():
    """markdown 代码块包裹的 JSON 应能解析"""
    result = parse_tool_call(
        '```json\n{"tool": "search_kb", "arguments": {"query": "x"}}\n```'
    )
    assert result is not None
    assert result.tool == "search_kb"


def test_json_without_tool_field_returns_none():
    """JSON 但没有 tool 字段 → 视为回答"""
    assert parse_tool_call('{"answer": "hello"}') is None


def test_tool_call_without_arguments():
    """没有 arguments 字段时默认为空 dict"""
    result = parse_tool_call('{"tool": "search_kb"}')
    assert result is not None
    assert result.arguments == {}
