"""
Agent 数据模型与工具调用解析

定义 Agent 循环中使用的数据结构：
  - ToolCall：解析后的工具调用
  - AgentResult：Agent 的最终返回结果

核心函数 parse_tool_call()：
  把 LLM 输出的文本解析为工具调用。
  LLM 的输出有两种可能：
    1. JSON 格式的工具调用 → 解析为 ToolCall
    2. 普通回答文本 → 返回 None（表示"这是最终回答"）
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# 匹配工具调用子对象：{"tool": "xxx", "arguments": {...}}
# 用于提取被误嵌在非法外层结构中的工具调用（缺键名等模型错误）
_TOOL_CALL_PATTERN = re.compile(
    r'\{"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)


@dataclass
class ToolCall:
    """
    一次工具调用。

    thought:     LLM 的思考（为什么调用这个工具）——用于 Trace 和调试
    tool:        工具名
    arguments:   工具参数（dict）
    """
    tool: str
    arguments: dict = field(default_factory=dict)
    thought: str = ""


@dataclass
class AgentResult:
    """
    Agent 一次执行的结果。

    answer:   最终回答文本
    completed:是否正常完成
    steps:    执行了多少步
    reason:   未完成的原因（completed=False 时）：
              "max_steps"   → 达到步数上限被截断
              "tool_errors" → 工具调用连续出错主动放弃
    """
    answer: str
    completed: bool = True
    steps: int = 0
    reason: str | None = None


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    解析 LLM 输出。

    判定规则：
      - 文本是合法的 JSON 且包含 "tool" 字段 → 工具调用
      - 其他情况 → 返回 None（视为最终回答）

    对格式错误的 JSON 宽容处理（真实世界的 LLM 输出总是"脏"的）：
      1. 模型可能用 ```json 代码块包裹 → 去掉包裹
      2. 模型可能先输出一大段内心独白再附 JSON（关闭 think 模式后常见）
         → 从混合文本中提取 {..} 子串再解析
      3. 整体解析失败 → 尝试子串提取，仍失败才视为最终回答

    Args:
        text: LLM 输出的原始文本

    Returns:
        ToolCall（是工具调用）或 None（是最终回答）
    """
    text = text.strip()

    # 尝试去掉 markdown 代码块包裹（```json ... ```）
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 尝试 1：整段作为 JSON 解析
    data = _parse_json(text)
    if data and "tool" in data:
        return _to_tool_call(data)

    # 尝试 2：截断的 JSON（小模型生成中断常见）——
    # 以 { 开头但整体不完整，逐次补 1~3 个 } 再试
    # 注意：不能用 "}" not in text 判断——嵌套对象内部的 } 会干扰判断。
    # 完整 JSON 补 } 后会解析失败（多一个括号），无害。
    if text.startswith("{") and '"tool"' in text:
        for n in range(1, 4):
            data = _parse_json(text + "}" * n)
            if data and "tool" in data:
                return _to_tool_call(data)

    # 尝试 3：从混合文本中提取 {..} 子串
    # 模型可能先输出思考/规划文本，最后附一个 JSON 工具调用
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        data = _parse_json(text[start:end + 1])
        if data and "tool" in data:
            return _to_tool_call(data)

    # 尝试 4：修复字符串值内部的裸引号后重试
    # （小模型在 JSON 字符串里用引号强调词时忘记转义，如 查找"microservice"，
    #   json.loads 会失败——把裸引号替换为中文引号）
    repaired = _repair_json_quotes(text)
    if repaired != text:
        data = _parse_json(repaired)
        if data and "tool" in data:
            return _to_tool_call(data)

    # 尝试 5：正则提取 tool 调用子对象
    # 模型可能把整个 {"tool": "...", "arguments": {...}} 误放为外层的值
    # （如 "thought": "...", {"tool": "search_kb", ...}——缺键名导致整体非法），
    # 但工具调用子对象本身是完整的，用正则提取。
    m = _TOOL_CALL_PATTERN.search(text)
    if m:
        try:
            args = json.loads(m.group(2))
            return ToolCall(tool=m.group(1), arguments=args, thought="")
        except json.JSONDecodeError:
            pass

    # 都失败 → 视为最终回答
    return None


def _parse_json(text: str) -> dict | None:
    """尝试解析 JSON，失败返回 None"""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _repair_json_quotes(text: str) -> str:
    """
    修复 JSON 字符串值内部的裸引号。

    小模型的常见错误：在字符串值里用引号强调词（如 查找"microservice"），
    但 JSON 语法要求字符串内部的引号必须转义为 \\"，裸引号会导致 json.loads 失败。

    修复策略：扫描文本，维护"是否在字符串内"状态——
      1. 字符串内的 \\" 是转义引号，保留
      2. 字符串内遇到裸引号：向后看，若后面是结构字符（, } ] :）则是字符串结束；
         否则是强调引号 → 替换为中文引号（左右交替，尽量保持可读性）
      3. 字符串外的引号是结构引号，保留
    """
    result = []
    in_string = False
    quote_count = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # 字符串内的转义（\\ 或 \" 等），原样保留两个字符
        if in_string and ch == "\\" and i + 1 < n:
            result.append(ch)
            result.append(text[i + 1])
            i += 2
            continue

        if ch == '"':
            if not in_string:
                # 进入字符串（结构引号）
                in_string = True
                result.append(ch)
            else:
                # 字符串内遇到引号：判断是字符串结束还是裸引号
                j = i + 1
                while j < n and text[j] in " \t\n":
                    j += 1
                if j < n and text[j] in ",}]:":
                    # 后跟结构字符 → 字符串结束
                    in_string = False
                    result.append(ch)
                else:
                    # 裸引号（强调用）→ 中文引号，左右交替
                    quote_count += 1
                    result.append("“" if quote_count % 2 == 1 else "”")
        else:
            result.append(ch)

        i += 1

    return "".join(result)


def _to_tool_call(data: dict) -> ToolCall:
    """
    把解析出的 dict 转为 ToolCall。

    容错：模型可能把 tool 字段写成嵌套对象
    （{"tool": {"name": "grep_code", "arguments": {...}}}——
    误解了调用示例的结构），需要提取 name 和 arguments。
    """
    raw_tool = data["tool"]
    arguments = data.get("arguments", {}) or {}

    if isinstance(raw_tool, dict):
        # 嵌套对象：{"name": "...", "arguments": {...}}
        arguments = arguments or raw_tool.get("arguments", {}) or {}
        raw_tool = raw_tool.get("name", "")

    return ToolCall(
        tool=str(raw_tool),
        arguments=arguments,
        thought=str(data.get("thought", "")),
    )
