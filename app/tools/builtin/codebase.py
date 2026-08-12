"""
代码库工具集（对齐 Claude Code 的 Grep / Read / Glob）

解决情境：研究报告需要对代码库做精确分析。
向量检索（search_kb）适合语义匹配文档，但对代码定位能力差。

三个工具：
  grep_code   按正则搜索代码内容（对齐 Grep）
  read_file   读取文件内容，带行号（对齐 Read）
  list_files  按模式列出文件（对齐 Glob）

安全设计（read_file 的路径穿越防护）：
  LLM 可能构造 "../../etc/passwd" 之类的路径——
  所有读取必须校验在代码库根目录内。
"""

import os
import re

from ..base import BaseTool, ToolSpec


class GrepCodeTool(BaseTool):
    """在代码库中按正则搜索文件内容（对齐 Claude Code 的 Grep）"""

    def __init__(self, codebase_dir: str):
        self.codebase_dir = codebase_dir

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="grep_code",
            description=(
                "在项目代码库中按正则表达式搜索文件内容，返回 文件路径:行号:匹配行。"
                "当你需要定位代码实现、查找函数/类定义、搜索特定 API 用法时调用。"
                "这是代码分析的入口——先 grep 找到相关文件，再 read_file 深入阅读。"
                "调用示例：{\"tool\": \"grep_code\", \"arguments\": {\"pattern\": \"def main\"}}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "正则表达式，如 'def process|class Gesture'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回的匹配行数（默认 20，避免刷屏）",
                        "default": 20,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        )

    async def run(self, pattern: str, max_results: int = 20) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"正则表达式无效: {e}"

        matches: list[str] = []
        for root, _, files in os.walk(self.codebase_dir):
            # 跳过常见非源码目录
            if any(part in root for part in (".git", "__pycache__", "node_modules")):
                continue
            for fn in files:
                if len(matches) >= max_results:
                    break
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            if regex.search(line):
                                # 统一用正斜杠（跨平台一致，LLM 和前端看到的路径格式统一）
                                rel = os.path.relpath(path, self.codebase_dir).replace(os.sep, "/")
                                matches.append(f"{rel}:{lineno}: {line.rstrip()[:150]}")
                                if len(matches) >= max_results:
                                    break
                except OSError:
                    continue
            if len(matches) >= max_results:
                break

        if not matches:
            # 无结果时给出"按目录分组的文件清单"：
            # 文件名/目录名本身携带语义（profiling→性能、config→配置），
            # 模型据此选择相关文件直接阅读——不依赖模型自觉猜测。
            return (
                f"未找到匹配 '{pattern}' 的内容。\n\n"
                f"请根据以下项目文件结构，选择与需求相关的文件直接 read_file 阅读：\n"
                f"{self._file_structure()}"
            )
        return "\n".join(matches)

    def _file_structure(self, max_per_dir: int = 8) -> str:
        """按目录分组列出文件（供 grep 无结果时引导阅读）"""
        dirs: dict[str, list[str]] = {}
        for root, _, fns in os.walk(self.codebase_dir):
            if any(part in root for part in (".git", "__pycache__", "node_modules")):
                continue
            rel = os.path.relpath(root, self.codebase_dir)
            label = "." if rel == "." else rel.replace(os.sep, "/")
            dirs.setdefault(label, [])
            for fn in sorted(fns):
                if len(dirs[label]) < max_per_dir:
                    dirs[label].append(fn)

        lines = []
        for d in sorted(dirs):
            files = ", ".join(dirs[d])
            lines.append(f"  {d}/: {files}" + ("..." if len(dirs[d]) == max_per_dir else ""))
        return "\n".join(lines)


class ReadFileTool(BaseTool):
    """读取文件内容，带行号（对齐 Claude Code 的 Read）"""

    def __init__(self, codebase_dir: str):
        self.codebase_dir = codebase_dir

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_file",
            description=(
                "读取代码库中某个文件的完整内容（带行号）。"
                "在 grep_code / list_files 定位到文件后调用，用于深入分析代码实现。"
                "路径使用相对代码库根目录的路径（如 src/main.py），"
                "禁止带 ./data/kb/ 等前缀。"
                "调用示例：{\"tool\": \"read_file\", \"arguments\": {\"path\": \"src/main.py\"}}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对代码库根目录的文件路径",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "最多读取的字符数（默认 4000，防止超长文件刷屏）",
                        "default": 4000,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def run(self, path: str, max_chars: int = 4000) -> str:
        # 路径容错：兼容相对路径（src/main.py）和完整路径
        # （./data/kb/xxx/source/src/main.py——search_kb 返回的格式）
        full_path = self._resolve_path(path)
        if full_path is None:
            return f"非法路径: {path}（只能读取代码库内的文件）"
        if not os.path.isfile(full_path):
            return f"文件不存在: {path}"

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(内容过长已截断)"

        # 带行号返回
        lines = content.splitlines()
        return "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))

    def _resolve_path(self, path: str) -> str | None:
        """
        解析文件路径（兼容两种格式），并校验在代码库根目录内。

        格式 1（推荐）：相对路径，如 src/main.py
        格式 2（容错）：完整路径，如 ./data/kb/xxx/source/src/main.py
                        （search_kb 返回的 metadata.source 格式）

        Returns:
            解析后的绝对路径；不在代码库内返回 None
        """
        codebase_root = os.path.realpath(self.codebase_dir)

        # 尝试 1：作为相对路径拼接
        full = os.path.realpath(os.path.join(codebase_root, path))
        if full.startswith(codebase_root):
            return full

        # 尝试 2：作为完整路径（已在代码库内）
        full = os.path.realpath(path)
        if full.startswith(codebase_root):
            return full

        return None


class ListFilesTool(BaseTool):
    """列出代码库文件（对齐 Claude Code 的 Glob）"""

    def __init__(self, codebase_dir: str):
        self.codebase_dir = codebase_dir

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_files",
            description=(
                "列出代码库中的文件。"
                "当你需要了解项目整体结构、有哪些文件时调用。"
                "支持通配符模式（如 '*.py'、'src/*'）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "文件通配符模式（默认 '*' 列出全部）",
                        "default": "*",
                    },
                },
                "additionalProperties": False,
            },
        )

    async def run(self, pattern: str = "*") -> str:
        import fnmatch

        files: list[str] = []
        for root, _, fns in os.walk(self.codebase_dir):
            if any(part in root for part in (".git", "__pycache__", "node_modules")):
                continue
            for fn in fns:
                rel = os.path.relpath(os.path.join(root, fn), self.codebase_dir)
                # 统一正斜杠（跨平台一致）
                rel = rel.replace(os.sep, "/")
                if fnmatch.fnmatch(rel, pattern):
                    files.append(rel)

        if not files:
            return f"没有匹配 '{pattern}' 的文件"
        return f"共 {len(files)} 个文件：\n" + "\n".join(sorted(files))
