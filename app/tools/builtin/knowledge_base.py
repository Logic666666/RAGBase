"""
知识库检索工具

将 RagService.search_docs() 包装为 Agent 可调用的工具。

和 RagService.answer_question() 的区别：
  - answer_question: 检索 + LLM 生成回答（给 /chat 用）
  - KnowledgeBaseTool: 只检索不生成（给 Agent 用，Agent 自己在循环里做生成）

设计选择：
  知识库名称在初始化时注入，而不是每次调用时传参。
  因为用户在使用 Agent 前已经选好了知识库，每次传参是多余的。
"""

import os

from ...services.rag import RagService
from ..base import BaseTool, ToolSpec


class KnowledgeBaseTool(BaseTool):
    """
    知识库检索工具。

    在选定的知识库中搜索与 query 相关的文档片段。
    Agent 拿到检索结果后自行决定下一步——是否需要再搜、还是够了可以生成回答。

    这是一个"只检索不做生成"的纯工具。
    """

    def __init__(self, rag_service: RagService, kb_name: str):
        """
        Args:
            rag_service: RagService 实例（共享 search_docs 方法）
            kb_name:     知识库名称（用户已选好，初始化时固定）
        """
        self.rag = rag_service
        self.kb_name = kb_name
        # 连续命中代码文件的次数：达到阈值后本工具"自我拒绝"，
        # 迫使模型查阅工具列表换工具（不点名其他工具，靠 description 自发现）
        self._code_file_hits = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_kb",
            description=(
                "在知识库中检索与 query 语义相关的文档片段（README、方案、笔记等文档）。"
                "用于获取项目背景、技术方案、历史决策等信息。"
                "返回片段按相关度排列，包含来源路径。"
                "注意：这是文档检索工具；分析代码实现请用 grep_code / read_file / list_files。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词。"
                                       "建议用核心概念词，不要用完整问句。"
                                       "如：'向量数据库 性能 对比' 而不是 "
                                       "'你能帮我对比一下向量数据库的性能吗'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回前 N 条最相关的文档片段（默认 4，最大 10）",
                        "default": 4,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def run(self, query: str, top_k: int = 4) -> str:
        """
        执行知识库检索。

        返回格式：
        [1] (source_path)
        文档内容...

        [2] (source_path)
        文档内容...

        Args:
            query: 搜索关键词
            top_k: 返回结果数量

        Returns:
            格式化的文本结果，每段带来源引用
        """
        docs = await self.rag.search_docs(self.kb_name, query, top_k)

        if not docs:
            return "知识库中没有找到与查询相关的内容。"

        parts = [
            f"[{i+1}] ({d['source']})\n{d['text']}"
            for i, d in enumerate(docs)
        ]
        result = "\n\n".join(parts)

        # 事实反馈：检索结果中包含代码文件时如实告知——
        # 不点名具体工具（工具选择依据见系统提示中的工具选择指南）
        code_exts = (".py", ".java", ".js", ".ts", ".go", ".cpp", ".c", ".h", ".rs")
        code_sources = [d["source"] for d in docs if d["source"].lower().endswith(code_exts)]

        # 连续命中代码文件达到阈值 → 本工具"自我拒绝"：
        # 说明该查询方向对文档检索无效，模型必须换工具（靠查阅工具列表自发现）。
        # 这是工具对自身能力的声明，不点名其他工具。
        if code_sources:
            self._code_file_hits += 1
            if self._code_file_hits >= 2:
                return (
                    "本工具（search_kb）已连续多次命中代码文件，"
                    "它是文档检索工具，对代码分析无效。"
                    "请停止使用本工具，查阅可用工具列表，"
                    "选择适合代码分析的其他工具。"
                )
            result += (
                "\n\n[提示] 检索结果中包含代码文件（如 "
                + ", ".join(os.path.basename(s) for s in code_sources[:2])
                + "）。向量检索对代码文件的内容覆盖有限，"
                "如需分析代码实现，请查阅可用工具列表选择合适工具。"
            )
        else:
            # 命中文档 → 重置计数（工具恢复可用）
            self._code_file_hits = 0

        return result
