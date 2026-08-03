"""
工具系统基础抽象

定义了工具系统的最小接口：
  - ToolSpec：工具的"身份证"（名字、描述、参数 schema）
  - ToolResult：工具执行的结构化结果（成功/错误标记）
  - BaseTool：所有工具必须继承的抽象类

一个工具 = 一个 spec（告诉 LLM 我能干什么）
         + 一个 run 方法（实际执行逻辑）

参数校验与异常兜底：
  BaseTool.execute() 是统一执行入口：
    1. 用 jsonschema 校验 LLM 传入的参数（LLM 输出不可信）
    2. 捕获 run() 的异常并转为结构化错误

设计参考（Claude Code）：
  Claude Code 用 tool_result + is_error 标记结构化返回错误，
  程序用字段判断而非解析文本。这里用 ToolResult.ok 承担同样职责。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jsonschema


@dataclass
class ToolResult:
    """
    工具执行的结构化结果。

    对应 Claude Code 的 tool_result 块（带 is_error 标记）：
      - ok=True  → 执行成功，content 是结果文本
      - ok=False → 执行失败（参数校验失败/内部异常），content 是错误信息

    设计要点：
      orchestrator 用 result.ok 判断成功/失败（字段判断），
      而不是 startswith("工具参数错误") 之类的字符串嗅探。
    """
    ok: bool
    content: str

    @classmethod
    def success(cls, content: str) -> "ToolResult":
        return cls(ok=True, content=content)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(ok=False, content=message)

    def to_message(self) -> str:
        """转成喂给 LLM 的文本内容"""
        return self.content


class ToolSpec:
    """
    工具规格描述——工具的"身份证"。

    name:        工具名，LLM 用它来指代这个工具（如 "search_kb"）
    description: 工具描述，LLM 用它来决定"什么时候该用这个工具"
                 写清楚"什么场景下调用"比写清楚"这个工具做什么"更重要
    parameters:  完整 JSON Schema 对象，定义工具参数的结构
                 遵循 OpenAI 工具定义的业界标准格式：
                 {
                     "type": "object",
                     "properties": {"query": {"type": "string"}},
                     "required": ["query"],
                     "additionalProperties": False
                 }
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
        }

    def to_dict(self) -> dict:
        """导出为 LLM 可读的字典格式（放在 system prompt 中）"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class BaseTool(ABC):
    """
    所有工具的基类。

    子类需要实现：
      spec:  工具的规格描述（告诉 LLM 我能干什么）
      run(): 工具的执行逻辑（实际做了什么）

    子类不需要实现 execute()——它由基类统一提供：
      参数校验（jsonschema）+ 异常兜底（错误转文本）
    """

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """工具的规格描述"""
        ...

    @abstractmethod
    async def run(self, **kwargs) -> str:
        """
        执行工具的实际逻辑。

        注意：这里可以抛异常，execute() 会兜住。
        参数由 LLM 根据 spec.parameters 生成，
        所以 parameters 的定义要准确，否则 LLM 会传错参数。

        返回字符串，直接追加到对话历史中给 LLM 看。
        如果需要返回结构化数据（如检索结果），
        在这里转成格式良好的文本。
        """
        ...

    # ──────────────────────────────────────────
    # 统一执行入口（模板方法）
    # ──────────────────────────────────────────

    async def execute(self, kwargs: dict) -> ToolResult:
        """
        统一的工具执行入口——orchestrator 只调这个方法。

        流程：
        1. 用 jsonschema 校验参数（LLM 生成的参数不可信）
        2. 调用子类的 run() 真正执行
        3. 捕获所有异常，转为结构化错误

        保证：这个方法永不抛异常。
        返回 ToolResult——用 ok 字段区分成功/失败，
        对应 Claude Code 的 tool_result + is_error 标记。
        """
        # 1. 参数校验
        error = self._validate_params(kwargs)
        if error:
            return ToolResult.error(
                f"工具参数错误: {error}。请检查参数后重试。"
            )

        # 2. 执行 + 异常兜底
        try:
            content = await self.run(**kwargs)
            return ToolResult.success(content)
        except Exception as e:
            return ToolResult.error(
                f"工具执行失败: {type(e).__name__}: {str(e)}。"
                f"请调整策略后重试。"
            )

    def _validate_params(self, kwargs: dict) -> str | None:
        """
        用 jsonschema 校验参数。

        spec.parameters 是完整 JSON Schema，
        jsonschema.validate(instance=kwargs, schema=parameters) 自动执行校验：
        - 缺少必填参数 → 报错
        - 参数类型不对 → 报错
        - 出现 schema 外的参数（additionalProperties: false）→ 报错

        Returns:
            错误描述字符串；参数合法返回 None
        """
        try:
            jsonschema.validate(
                instance=kwargs,
                schema=self.spec.parameters,
            )
            return None
        except jsonschema.ValidationError as e:
            # e.message 是 jsonschema 生成的准确错误描述
            return f"{e.message} (path: {list(e.path)})"
