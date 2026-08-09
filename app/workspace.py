"""
工作区（对齐 Claude Code 的 filesystem as long-term memory）

解决情境：研究报告任务中，Agent 发现的中间信息
（关键资料、分析要点、草稿）不能全留在消息历史——
上下文是有限资源，中间产物必须外置到文件系统。

Claude Code 的做法（官方 best-practices）：
  "shorter context leads to faster and smarter operation"——
  把重要信息写成文件，需要时再读，而不是常驻上下文。

设计：
  Workspace      抽象接口（实现可替换）
  FileWorkspace  文件系统实现：data/sessions/{session_id}/notes/
                 （按会话隔离，每个研究报告任务有自己的笔记）
"""

import os
from abc import ABC, abstractmethod


class Workspace(ABC):
    """
    工作区抽象接口。

    上层（note_take 工具）只依赖本接口：
      save_note / read_note / list_notes
    """

    @abstractmethod
    async def save_note(self, name: str, content: str) -> str:
        """保存笔记，返回笔记路径"""
        ...

    @abstractmethod
    async def read_note(self, name: str) -> str:
        """读取笔记内容"""
        ...

    @abstractmethod
    async def list_notes(self) -> list[str]:
        """列出所有笔记名"""
        ...


class FileWorkspace(Workspace):
    """
    文件系统工作区实现。

    目录结构：{base_dir}/notes/
    """

    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: 工作区根目录（按会话隔离，如 data/sessions/{session_id}）
        """
        self.notes_dir = os.path.join(base_dir, "notes")
        os.makedirs(self.notes_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        # 笔记名做安全化：只允许字母数字下划线（防路径穿越）
        safe_name = "".join(c for c in name if c.isalnum() or c in "_-.")
        if not safe_name:
            safe_name = "note"
        return os.path.join(self.notes_dir, f"{safe_name}.md")

    async def save_note(self, name: str, content: str) -> str:
        path = self._path(name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    async def read_note(self, name: str) -> str:
        path = self._path(name)
        if not os.path.exists(path):
            return f"笔记 '{name}' 不存在"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    async def list_notes(self) -> list[str]:
        if not os.path.isdir(self.notes_dir):
            return []
        return sorted(
            fn[:-3] for fn in os.listdir(self.notes_dir) if fn.endswith(".md")
        )
