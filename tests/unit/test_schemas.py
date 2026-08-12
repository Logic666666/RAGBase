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


def test_mixed_text_with_tool_call():
    """
    模型先输出独白再附 JSON（关闭 think 模式后的常见行为）
    应从混合文本中提取出工具调用。
    """
    text = (
        "好的，我现在需要回答用户关于权衡的问题。"
        "首先我应该调用工具检索知识库。"
        '{"thought": "先搜索", "tool": "search_kb", '
        '"arguments": {"query": "手势识别 权衡"}}'
    )
    result = parse_tool_call(text)
    assert result is not None
    assert result.tool == "search_kb"
    assert result.arguments == {"query": "手势识别 权衡"}


def test_mixed_text_is_still_final_answer_without_json():
    """混合文本但没有 JSON 工具调用 → 视为最终回答"""
    text = "根据我的分析，这个项目使用了 MediaPipe 实现手势识别。"
    assert parse_tool_call(text) is None


def test_truncated_json_repaired():
    """
    截断的 JSON（缺结尾 }）应被修复解析。
    小模型生成中断时常输出不完整 JSON，若被当作最终回答会返回垃圾文本。
    """
    text = '{"thought": "需要读取代码", "tool": "read_file", "arguments": {"path": "main.py"'
    result = parse_tool_call(text)
    assert result is not None
    assert result.tool == "read_file"
    assert result.arguments == {"path": "main.py"}


def test_bare_quotes_in_string_repaired():
    """
    字符串值内部的裸引号（模型忘记转义）应被修复。
    真实案例：thought 里写 查找"microservice" 而未转义，json.loads 直接失败。
    """
    text = (
        '{"thought": "查找"microservice"或"containerization"相关关键词。", '
        '"tool": "grep_code", "arguments": {"pattern": "micro"}}'
    )
    result = parse_tool_call(text)
    assert result is not None
    assert result.tool == "grep_code"
    assert result.arguments == {"pattern": "micro"}
    # thought 修复后保留内容（裸引号 → 中文引号）
    assert "microservice" in result.thought


def test_escaped_quotes_not_damaged():
    """合法的转义引号 \\" 不应被修复逻辑破坏"""
    text = '{"thought": "他说 \\"你好\\"", "tool": "search_kb", "arguments": {"query": "x"}}'
    result = parse_tool_call(text)
    assert result is not None
    assert result.tool == "search_kb"
    assert "你好" in result.thought


def test_tool_field_as_nested_object():
    """
    模型把 tool 字段写成嵌套对象（误解调用示例结构）时，
    应提取 name 和嵌套的 arguments。
    真实案例：{"tool": {"name": "grep_code", "arguments": {...}}}。
    """
    text = (
        '{"thought": "grep_code 工具可用", '
        '"tool": {"name": "grep_code", "arguments": {"pattern": "micro"}}}'
    )
    result = parse_tool_call(text)
    assert result is not None
    assert result.tool == "grep_code"
    assert result.arguments == {"pattern": "micro"}


def test_nested_object_truncated_json():
    """
    嵌套 tool 对象 + 截断 JSON 的组合修复：
    内部有 } 不应干扰截断检测（真实 session 案例）。
    """
    text = (
        '{"thought": "grep_code 工具不可用", '
        '"tool": {"name": "search_kb", "arguments": {"query": "微服务"}}'
    )
    result = parse_tool_call(text)
    assert result is not None
    assert result.tool == "search_kb"
    assert result.arguments == {"query": "微服务"}


def test_missing_key_tool_call_extracted():
    """
    模型把整个工具调用对象误嵌为外层值（缺键名）时，
    用正则提取内部完整的 {"tool": ..., "arguments": {...}}。
    真实案例："thought": "...", {"tool": "search_kb", "arguments": {...}}。
    """
    text = (
        '{"thought": "需要先检索技术文档", '
        '{"tool": "search_kb", "arguments": {"query": "架构设计"}}}'
    )
    result = parse_tool_call(text)
    assert result is not None
    assert result.tool == "search_kb"
    assert result.arguments == {"query": "架构设计"}
