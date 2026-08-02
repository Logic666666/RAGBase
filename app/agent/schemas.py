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

    对格式错误的 JSON 宽容处理：
      模型可能输出前后带杂文本的 JSON（如 ```json 包裹），
      这里先做基础的清理再尝试解析。

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

    # 不是 JSON 对象 → 不是工具调用
    if not (text.startswith("{") and text.endswith("}")):
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # JSON 解析失败 → 视为最终回答（可能有模型输出瑕疵）
        return None

    # 必须有 "tool" 字段才是工具调用
    if "tool" not in data:
        return None

    return ToolCall(
        tool=str(data["tool"]),
        arguments=data.get("arguments", {}) or {},
        thought=str(data.get("thought", "")),
    )
