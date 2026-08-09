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

        # 引导换工具：如果检索到的都是代码文件（.py/.java 等），
        # 说明这是代码分析任务——向量检索对代码命中差，
        # 明确提示模型改用 grep_code / read_file 分析实现。
        # （小模型的工具选择能力弱，prompt 抽象规则不如具体提示有效）
        code_exts = (".py", ".java", ".js", ".ts", ".go", ".cpp", ".c", ".h", ".rs")
        if all(d["source"].lower().endswith(code_exts) for d in docs):
            result += (
                "\n\n[提示] 以上片段来自代码文件。"
                "若要分析代码实现，请使用 grep_code 定位相关文件、"
                "read_file 读取代码，而不是继续 search_kb。"
            )

        return result
