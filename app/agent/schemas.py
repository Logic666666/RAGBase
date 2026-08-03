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
from dataclasses import dataclass, field
from typing import Optional


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
    completed:是否正常完成（False 表示达到 max_steps 被截断）
    steps:    执行了多少步
    """
    answer: str
    completed: bool = True
    steps: int = 0


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

    # 尝试 2：从混合文本中提取 {..} 子串
    # 模型可能先输出思考/规划文本，最后附一个 JSON 工具调用
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        data = _parse_json(text[start:end + 1])
        if data and "tool" in data:
            return _to_tool_call(data)

    # 都失败 → 视为最终回答
    return None


def _parse_json(text: str) -> dict | None:
    """尝试解析 JSON，失败返回 None"""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _to_tool_call(data: dict) -> ToolCall:
    """把解析出的 dict 转为 ToolCall"""
    return ToolCall(
        tool=str(data["tool"]),
        arguments=data.get("arguments", {}) or {},
        thought=str(data.get("thought", "")),
    )
