"""
RAG（检索增强生成）问答服务

核心流程：
  用户提问 → 向量检索（VectorStore.similarity_search）→
  构建上下文 → LLM 生成回答（OllamaChatClient.chat）→ 返回答案+来源
"""

import os

from ..core.config import Settings
from ..infrastructure.vector_store import VectorStore
from ..infrastructure.llm_client import OllamaChatClient, Message

# 系统提示词
SYS_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "Cite sources as file paths if relevant. If the answer is not in the context, say you don't know."
)


class RagService:
    """
    RAG 问答服务。

    将基础设施层的组建组合成一条完整的问答管线：
    1. VectorStore → 检索相关文档
    2. OllamaChatClient → 基于检索结果生成回答
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vs = VectorStore(settings)
        self.llm = OllamaChatClient(
            base_url=settings.ollama_base_url,
            model=settings.chat_model,
            temperature=0.2,
        )

    def _kb_vector_dir(self, name: str) -> str:
        """获取知识库的向量存储目录"""
        return os.path.join(self.settings.data_dir, "vectorstore", name)

    async def answer_question(
        self, kb: str, question: str, top_k: int = 4
    ) -> tuple[str, list[dict]]:
        """
        执行 RAG 问答。

        流程：
        1. 向量检索：在知识库中搜索与问题最相关的文档块
        2. 构建上下文：将检索结果格式化为 LLM 可读的上下文
        3. LLM 生成：将上下文 + 问题发给模型，生成最终回答
        4. 整理来源：返回答案和引用来源

        参数：
            kb:       知识库名称
            question: 用户问题
            top_k:    返回最相似文档数

        返回：
            (answer_text, [{source, snippet}, ...])
        """
        # 1. 检索
        results = await self.vs.similarity_search(
            self._kb_vector_dir(kb),
            question,
            top_k,
        )

        # 2. 构建上下文
        # results = [(text, metadata, distance), ...]
        context_parts = [
            f"[{i+1}] ({meta.get('source', '')})\n{text}"
            for i, (text, meta, _) in enumerate(results)
        ]
        context = "\n\n".join(context_parts)

        # 3. LLM 生成
        messages = [
            Message(role="system", content=SYS_PROMPT),
            Message(
                role="user",
                content=f"Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
        answer = await self.llm.chat(messages)

        # 4. 整理来源
        sources = [
            {"source": meta.get("source", ""), "snippet": text[:300]}
            for text, meta, _ in results
        ]

        return answer, sources
