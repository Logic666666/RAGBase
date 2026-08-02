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

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_kb",
            description=(
                "在项目的知识库中搜索与 query 相关的文档片段。"
                "这是回答任何项目问题的唯一信息源，"
                "回答任何关于项目、文档、代码、技术方案的问题前必须先调用本工具。"
                "返回的文档片段按相关度从高到低排列，每个片段包含来源文件路径。"
                "如果一次检索结果不完整，可以换关键词多次调用。"
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
        return "\n\n".join(parts)
