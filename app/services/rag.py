"""
RAG 问答服务

提供两层 API：
  1. search_docs()  — 共享检索方法，纯检索不做生成
                      → 供 /chat 和 Agent 的 KnowledgeBaseTool 共同使用
  2. answer_question() — 薄封装：检索 + 一轮 LLM 生成
                      → 供 /chat 端点使用

不维护两套检索逻辑，一个 search_docs 两个使用方。
"""

import os

from ..core.config import Settings
from ..infrastructure.vector_store import VectorStore
from ..infrastructure.llm_client import OllamaChatClient, Message


SYS_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "Cite sources as file paths if relevant. If the answer is not in the context, say you don't know."
)


class RagService:
    """
    RAG 问答服务。

    职责：
    - search_docs：  在知识库中检索相关文档（纯检索，不做 LLM 生成）
                      被 /chat 和 Agent KnowledgeBaseTool 共享
    - answer_question：检索 + 一轮 LLM 生成（薄封装）
                      仅供 /chat 端点使用

    设计原则：
    一个能力（检索）只有一个来源（search_docs），
    不同的消费方（/chat、Agent）各自按需使用。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vs = VectorStore(settings)
        self.llm = OllamaChatClient(
            base_url=settings.ollama_base_url,
            model=settings.chat_model,
            temperature=0.2,
            think=settings.llm_think,
            max_tokens=settings.llm_max_tokens,
        )

    # ──────────────────────────────────────────
    # 共享检索方法
    # ──────────────────────────────────────────

    async def search_docs(
        self, kb: str, query: str, top_k: int = 4
    ) -> list[dict]:
        """
        在知识库中检索与 query 相关的文档片段。

        这是所有检索需求的统一入口：
        - /chat 用：作为 answer_question 的第一步
        - Agent 用：作为 KnowledgeBaseTool 的核心逻辑

        返回结构化结果，而不是 LLM 生成的回答：
        [{ "text": "文档片段", "source": "文件路径", "score": 0.95 }, ...]

        Args:
            kb:      知识库名称
            query:   搜索关键词
            top_k:   返回前 N 条最相关文档

        Returns:
            按相似度降序排列的文档列表
        """
        results = await self.vs.similarity_search(
            self._kb_vector_dir(kb), query, top_k
        )

        # results = [(text, metadata, distance), ...]
        return [
            {
                "text": text,
                "source": meta.get("source", ""),
                "score": 1 - dist,  # distance→similarity 转换
            }
            for text, meta, dist in results
        ]

    # ──────────────────────────────────────────
    # RAG 问答封装
    # ──────────────────────────────────────────

    async def answer_question(
        self, kb: str, question: str, top_k: int = 4
    ) -> tuple[str, list[dict]]:
        """
        单轮 RAG 问答：检索 → 构建上下文 → LLM 生成。

        这是 /chat 端点的后端逻辑。
        不走 ReAct 循环，适合直接问答场景。

        返回：
            (answer_text, [{source, snippet}, ...])
        """
        # 1. 检索
        docs = await self.search_docs(kb, question, top_k)

        # 2. 构建上下文
        context_parts = [
            f"[{i+1}] ({d['source']})\n{d['text']}"
            for i, d in enumerate(docs)
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
            {"source": d["source"], "snippet": d["text"][:300]}
            for d in docs
        ]

        return answer, sources

    # ──────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────

    def _kb_vector_dir(self, name: str) -> str:
        """获取知识库的向量存储目录"""
        return os.path.join(self.settings.data_dir, "vectorstore", name)
