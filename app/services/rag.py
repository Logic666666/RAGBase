import os
from typing import List, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from ..settings import Settings
from .vectorstore import VectorStore


SYS_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "Cite sources as file paths if relevant. If the answer is not in the context, say you don't know."
)


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vs = VectorStore(settings)

    def _llm(self) -> ChatOllama:
        return ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=self.settings.deepseek_model,
            temperature=0.2,
        )

    def _kb_vector_dir(self, name: str) -> str:
        return os.path.join(self.settings.data_dir, "vectorstore", name)

    def answer_question(self, kb: str, question: str, top_k: int = 4) -> tuple[str, List[dict]]:
        retriever = self.vs.as_retriever(self._kb_vector_dir(kb), top_k)
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([f"[{i+1}] ({d.metadata.get('source','')})\n{d.page_content}" for i, d in enumerate(docs)])

        messages = [
            SystemMessage(content=SYS_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]

        resp = self._llm().invoke(messages)
        answer = resp.content if hasattr(resp, "content") else str(resp)
        sources = [{"source": d.metadata.get("source", ""), "snippet": d.page_content[:300]} for d in docs]
        return answer, sources


