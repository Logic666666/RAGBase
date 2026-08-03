"""
工具注册中心

管理所有可用工具的注册、发现和执行。

职责：
  1. register():  把工具注册进来（在应用启动时完成）
  2. schemas():   生成所有工具的描述列表（放在 system prompt 里给 LLM 看）
  3. execute():   根据工具名和参数执行对应的工具

"""

from typing import Any

from .base import BaseTool, ToolResult


class ToolRegistry:
    """
    工具注册中心。

    用法：
      registry = ToolRegistry()
      registry.register(MyTool())
      registry.register(OtherTool())

      # 生成给 LLM 看的工具列表
      schemas = registry.schemas()  # → [{name, description, parameters}, ...]

      # 执行工具
      result = await registry.execute("my_tool", {"arg": "value"})
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        注册一个工具。

        工具名是唯一标识。如果重复注册后面的覆盖前面的。
        """
        self._tools[tool.spec.name] = tool

    def schemas(self) -> list[dict]:
        """
        生成所有工具的描述列表，给 LLM 看。

        返回格式：
        [
            {
                "name": "search_kb",
                "description": "在知识库中搜索文档...",
                "parameters": {"query": {"type": "string", ...}}
            },
            ...
        ]

        这个列表会被序列化到 system prompt 中，
        LLM 根据这些描述决定调用哪个工具。
        """
        return [tool.spec.to_dict() for tool in self._tools.values()]

    def get(self, name: str) -> BaseTool | None:
        """按名称获取工具"""
        return self._tools.get(name)

    async def execute(self, name: str, kwargs: dict[str, Any]) -> ToolResult:
        """
        执行指定的工具。

        Args:
            name:   工具名
            kwargs: 工具参数（由 LLM 生成，所以参数的 schema 要准确）

        Returns:
            ToolResult——ok 标记成功/失败。
            工具不存在、参数不合法、执行出错时，
            返回结构化错误而不是抛异常——LLM 需要看到错误来调整策略。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(
                f"工具 '{name}' 不存在。"
                f"可用工具: {', '.join(self._tools.keys())}。"
            )

        # BaseTool.execute() 内部完成参数校验 + 异常兜底
        return await tool.execute(kwargs)
