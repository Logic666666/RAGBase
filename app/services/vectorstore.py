"""
向量存储服务模块

该模块提供了基于Chroma向量数据库的文档存储和检索功能，
使用Ollama提供的嵌入模型将文本转换为向量表示。
"""

import os
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from ..settings import Settings


class VectorStore:
    """
    向量存储服务类，负责文档的向量化存储和相似性检索
    
    该类封装了Chroma向量数据库的操作，提供了文档添加和检索功能，
    使用Ollama的嵌入模型将文本转换为向量表示。
    """
    
    def __init__(self, settings: Settings) -> None:
        """
        初始化向量存储服务
        
        Args:
            settings: 应用程序设置对象，包含Ollama服务地址和模型配置
        """
        self.settings = settings

    def _get_safe_collection_name(self, persist_dir: str) -> str:
        """
        生成符合ChromaDB规范的集合名称
        
        ChromaDB要求：
        - 名称必须以字母或数字开头和结尾
        - 只能包含 [a-zA-Z0-9._-] 字符
        - 长度在 3-512 之间
        
        Args:
            persist_dir: 向量数据库的持久化存储目录路径
            
        Returns:
            符合ChromaDB规范的集合名称
        """
        import re
        import hashlib
        
        dir_name = os.path.basename(persist_dir)
        # 如果目录名包含非ASCII字符，使用哈希值
        if not dir_name.isascii():
            # 使用哈希值确保唯一性和安全性
            hash_value = hashlib.md5(dir_name.encode('utf-8')).hexdigest()[:8]
            safe_collection_name = f"kb_{hash_value}"
        else:
            # 移除非字母数字字符，但保留字母数字和下划线
            safe_collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', dir_name)
            # 移除开头和结尾的下划线
            safe_collection_name = safe_collection_name.strip('_')
            # 如果处理后为空或太短，使用哈希值
            if not safe_collection_name or len(safe_collection_name) < 2:
                hash_value = hashlib.md5(dir_name.encode('utf-8')).hexdigest()[:8]
                safe_collection_name = f"kb_{hash_value}"
            else:
                safe_collection_name = f"kb_{safe_collection_name}"
        
        # 确保名称以字母或数字开头和结尾（ChromaDB要求）
        safe_collection_name = re.sub(r'^[^a-zA-Z0-9]+', '', safe_collection_name)
        safe_collection_name = re.sub(r'[^a-zA-Z0-9]+$', '', safe_collection_name)
        
        # 确保名称长度符合要求（3-512字符）
        if len(safe_collection_name) < 3:
            hash_value = hashlib.md5(dir_name.encode('utf-8')).hexdigest()[:8]
            safe_collection_name = f"kb{hash_value}"
        elif len(safe_collection_name) > 512:
            # 如果太长，截断并添加哈希值确保唯一性
            hash_value = hashlib.md5(dir_name.encode('utf-8')).hexdigest()[:8]
            safe_collection_name = safe_collection_name[:500] + hash_value
        
        return safe_collection_name

    def _embeddings(self) -> OllamaEmbeddings:
        """
        创建Ollama嵌入模型实例（专门用于文本向量化）
        
        重要改进：现在可以独立配置嵌入模型，不受问答模型影响！
        
        嵌入模型配置说明：
        - 基础URL：从settings.ollama_base_url获取，默认值为"http://localhost:11434"
        - 模型名称：优先使用settings.embedding_model，后备使用settings.deepseek_model
        - 配置值来自app.settings.Settings类，可通过环境变量设置：
          * OLLAMA_BASE_URL：Ollama服务地址
          * EMBEDDING_MODEL：嵌入模型名称（新配置项）
          * DEEPSEEK_MODEL：后备兼容配置项
        
        模型切换说明：
        现在可以独立配置嵌入模型，不受问答模型影响！您可以：
        1. 下载专门的嵌入模型（推荐nomic-embed-text）
        2. 设置EMBEDDING_MODEL环境变量
        3. 保持问答模型使用其他模型（如deepseek-r1）
        
        专用嵌入模型推荐：
        - nomic-embed-text：专门优化的文本嵌入模型，性能最佳
        - mxbai-embed-large：多语言嵌入模型，支持中文
        - snowflake-arctic-embed：高性能嵌入模型
        
        与问答模型的区别：
        - 嵌入模型：专门将文本转换为向量（语义理解）
        - 问答模型：用于对话生成和推理（文本生成）
        - 两者可以独立配置，互不干扰！
        
        Returns:
            配置好的OllamaEmbeddings实例，用于将文本转换为向量表示
        """
        # 优先使用新的embedding_model配置，如果没有则使用旧的deepseek_model作为后备
        model_name = self.settings.embedding_model or self.settings.deepseek_model
        return OllamaEmbeddings(
            base_url=self.settings.ollama_base_url,
            model=model_name,
        )

    def add_documents(self, persist_dir: str, docs: List[Tuple[str, dict]]):
        """
        将文档添加到向量存储中
        
        处理流程：
        1. 提取文本内容和元数据
        2. 创建或加载Chroma向量数据库
        3. 将文本和对应的元数据添加到数据库
        4. 持久化存储到指定目录
        
        Args:
            persist_dir: 向量数据库的持久化存储目录路径
            docs: 文档列表，每个文档是(文本内容, 元数据字典)的元组
        """
        if not docs:
            return
            
        # 分离文本内容和元数据
        texts = [t for t, _ in docs]
        metas = [m for _, m in docs]
        
        # 创建Chroma向量存储实例
        # Chroma会自动调用embedding_function将文本转换为向量
        # 向量化过程：文本 -> 嵌入模型 -> 向量表示 -> 向量数据库存储
        # 生成安全的集合名称（符合ChromaDB规范：必须以字母或数字开头和结尾）
        safe_collection_name = self._get_safe_collection_name(persist_dir)
        
        vs = Chroma(
            collection_name=safe_collection_name,
            embedding_function=self._embeddings(),
            persist_directory=persist_dir
        )
        
        # 添加文本到向量存储
        # Chroma会自动处理向量化：对每个文本调用嵌入模型生成向量，然后存储
        vs.add_texts(texts=texts, metadatas=metas)
        
        # 持久化存储到磁盘
        # 数据存储位置：persist_directory指定的目录
        vs.persist()

    def as_retriever(self, persist_dir: str, top_k: int):
        """
        创建向量检索器
        
        从持久化存储中加载向量数据库，并配置为检索器，
        用于相似性搜索和文档检索。
        
        Args:
            persist_dir: 向量数据库的持久化存储目录路径
            top_k: 返回最相似的文档数量
            
        Returns:
            配置好的向量检索器，可用于相似性搜索
        """
        # 加载已存在的向量存储
        # Chroma会从persist_directory读取之前存储的向量数据
        # 使用与add_documents相同的集合名称生成逻辑，确保能正确加载
        safe_collection_name = self._get_safe_collection_name(persist_dir)
        vs = Chroma(
            collection_name=safe_collection_name,
            embedding_function=self._embeddings(),
            persist_directory=persist_dir
        )
        
        # 配置检索器，设置返回结果数量
        # 检索原理：将查询文本向量化，然后在向量空间中搜索相似文档
        # 相似度计算：基于向量距离（如余弦相似度）
        # 返回结果：按相似度排序的前top_k个文档
        return vs.as_retriever(search_kwargs={"k": top_k})

