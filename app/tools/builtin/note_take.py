"""
笔记工具（对齐 Claude Code 的 filesystem as memory）

解决情境：研究报告任务中，Agent 收集到的关键信息
如果不外置，会持续占用上下文。

工作流：
  收集到重要发现 → note_take 写入工作区笔记
  → 后续轮次需要时 → read_note 读取
  → 最终 assemble_report（Phase 4）从笔记组装报告

为什么笔记外置而不是留在上下文：
  上下文是有限资源（Claude Code："shorter context leads to
  faster and smarter operation"）——结论留笔记，细节按需读。
"""

from ...workspace import Workspace
from ..base import BaseTool, ToolSpec


class NoteTakeTool(BaseTool):
    """把关键发现写入工作区笔记（对齐 Claude Code 的 filesystem as memory）"""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="note_take",
            description=(
                "将关键发现写入工作区笔记（markdown 格式）。"
                "在研究过程中，当你获得重要信息、分析结论、待办要点时调用，"
                "避免信息积累在对话里。"
                "笔记按名称保存，后续可用 read_note 读取。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "笔记名（如 '向量数据库对比'）",
                    },
                    "content": {
                        "type": "string",
                        "description": "笔记内容（markdown，简洁记录要点）",
                    },
                },
                "required": ["name", "content"],
                "additionalProperties": False,
            },
        )

    async def run(self, name: str, content: str) -> str:
        path = await self.workspace.save_note(name, content)
        return f"笔记已保存: {path}"


class ReadNoteTool(BaseTool):
    """读取工作区笔记内容"""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_note",
            description=(
                "读取工作区中已保存的笔记内容。"
                "当你需要回顾之前的研究发现、汇总笔记时调用。"
                "可用 list_notes 查看有哪些笔记。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "笔记名（与 note_take 保存时一致）",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        )

    async def run(self, name: str) -> str:
        return await self.workspace.read_note(name)


class ListNotesTool(BaseTool):
    """列出工作区所有笔记"""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_notes",
            description=(
                "列出工作区中已保存的所有笔记名。"
                "当你需要了解自己记录了什么、回顾研究进度时调用。"
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    async def run(self) -> str:
        notes = await self.workspace.list_notes()
        if not notes:
            return "工作区暂无笔记。"
        return "已有笔记：\n" + "\n".join(f"- {n}" for n in notes)
