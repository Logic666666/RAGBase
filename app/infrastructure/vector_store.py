"""
向量存储模块

基于 ChromaDB 原生 SDK 的文档向量化存储与相似性检索。
不依赖 LangChain 的 Chroma。

核心流程：
  add_documents:     文本 → EmbeddingClient → 向量 → ChromaDB 存储
  similarity_search: 查询 → EmbeddingClient → 向量 → ChromaDB 搜索 → 返回结果

设计原则：
  - 不在 ChromaDB 侧自动调用 embedding，而是客户端算好向量再传入
  - 这样每个环节都可观测、可控制、可替换
"""

import os
import re
import hashlib
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from ..core.config import Settings
from .embeddings import OllamaEmbeddingClient


class VectorStore:
    """
    向量存储服务。

    封装了 ChromaDB 的核心操作（增、查、删），
    使用自实现的 OllamaEmbeddingClient 做文本向量化。

    ChromaDB 支持传一个 embedding function 让它自动调。
    但这样会"黑盒化" embedding 过程：
    - 不知道 embedding 何时算的、算得对不对
    - 没法加缓存（重复文本反复算）
    - 没法在中间加日志或监控
    - 换向量库时要重新配置 embedding

    故选择"客户端算好再传"的模式。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_client = OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
        )


    # 公开接口：添加文档
    async def add_documents(
        self, persist_dir: str, docs: list[tuple[str, dict]]
    ) -> None:
        """
        将文档添加到向量存储。

        流程：
        1. 分离文本和元数据
        2. 用 EmbeddingClient 将文本转为向量 ← 自己做，不假手 Chroma
        3. 用 ChromaDB 存储向量 + 原文 + 元数据

        Args:
            persist_dir: ChromaDB 持久化目录
            docs:        [(文本内容, 元数据字典), ...] 的列表
        """
        if not docs:
            return

        texts, metadatas = zip(*docs)

        # 核心：调 embedding
        embeddings = await self.embedding_client.embed_documents(list(texts))

        # 准备 ChromaDB 数据
        collection = self._get_or_create_collection(persist_dir)
        ids = [
            f"doc_{i}_{hash(text)}"
            for i, text in enumerate(texts)
        ]

        collection.add(
            embeddings=embeddings,        # 算好的向量
            documents=list(texts),        # 原文——ChromaDB 存一份方便查看
            metadatas=list(metadatas),    # 元数据
            ids=ids,
        )

    # 公开接口：相似度搜索
    async def similarity_search(
        self,
        persist_dir: str,
        query: str,
        top_k: int = 4,
    ) -> list[tuple[str, dict, float]]:
        """
        在向量存储中搜索与 query 语义相似的文档。

        流程：
        1. 把 query 转成向量
        2. 用 ChromaDB 搜索最相似的 top_k 个向量
        3. 返回 (原文, 元数据, 相似度分数)

        Args:
            persist_dir: ChromaDB 持久化目录
            query:       查询文本
            top_k:       返回前 N 条最相似的结果

        Returns:
            [(text, metadata, distance), ...]
            distance 越小表示越相似（余弦距离）
        """
        # 计算 query 的向量
        query_embedding = await self.embedding_client.embed_query(query)

        collection = self._get_or_create_collection(persist_dir)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # 整理结果
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return list(zip(documents, metadatas, distances))

    # 公开接口：删除集合
    def delete_collection(self, persist_dir: str) -> None:
        """删除整个向量集合（对应删除知识库时清理向量数据）"""
        collection_name = self._safe_collection_name(persist_dir)
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(collection_name)
        except ValueError:
            # 集合不存在时 ChromaDB 会抛 ValueError，忽略
            pass


    # 内部方法
    def _get_or_create_collection(self, persist_dir: str):
        """获取或创建 ChromaDB 集合"""
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection_name = self._safe_collection_name(persist_dir)

        try:
            return client.get_collection(name=collection_name)
        except (ValueError, NotFoundError):
            return client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def _safe_collection_name(self, persist_dir: str) -> str:
        """
        生成符合 ChromaDB 规范的集合名称。

        ChromaDB 要求：
        - 字母或数字开头/结尾
        - 只能包含 [a-zA-Z0-9._-]
        - 长度 3-512 字符
        """
        dir_name = os.path.basename(persist_dir)

        # 非 ASCII → 用哈希
        if not dir_name.isascii():
            hash_val = hashlib.md5(dir_name.encode()).hexdigest()[:8]
            return f"kb_{hash_val}"

        # 清理非法字符
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", dir_name).strip("_")
        if not safe or len(safe) < 2:
            hash_val = hashlib.md5(dir_name.encode()).hexdigest()[:8]
            return f"kb_{hash_val}"

        safe = f"kb_{safe}"

        # 确保以字母/数字开头结尾
        safe = re.sub(r"^[^a-zA-Z0-9]+", "", safe)
        safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)

        if len(safe) < 3:
            hash_val = hashlib.md5(dir_name.encode()).hexdigest()[:8]
            return f"kb{hash_val}"
        if len(safe) > 512:
            hash_val = hashlib.md5(dir_name.encode()).hexdigest()[:8]
            safe = safe[:500] + hash_val

        return safe
