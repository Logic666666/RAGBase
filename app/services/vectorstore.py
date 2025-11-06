import os
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from ..settings import Settings


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            base_url=self.settings.ollama_base_url,
            model=self.settings.deepseek_model,
        )

    def add_documents(self, persist_dir: str, docs: List[Tuple[str, dict]]):
        if not docs:
            return
        texts = [t for t, _ in docs]
        metas = [m for _, m in docs]
        vs = Chroma(collection_name="kb", embedding_function=self._embeddings(), persist_directory=persist_dir)
        vs.add_texts(texts=texts, metadatas=metas)
        vs.persist()

    def as_retriever(self, persist_dir: str, top_k: int):
        vs = Chroma(collection_name="kb", embedding_function=self._embeddings(), persist_directory=persist_dir)
        return vs.as_retriever(search_kwargs={"k": top_k})


