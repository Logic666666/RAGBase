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
        """
        创建聊天模型实例（用于问答生成）
        
        模型配置说明：
        - 优先使用新的chat_model配置，如果没有则使用旧的deepseek_model作为后备
        - 这样可以独立配置问答模型，不受嵌入模型影响
        
        Returns:
            配置好的ChatOllama实例，用于对话生成
        """
        # 优先使用新的chat_model配置，如果没有则使用旧的deepseek_model作为后备
        model_name = self.settings.chat_model or self.settings.deepseek_model
        return ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=model_name,
            temperature=0.2,
        )

    def _kb_vector_dir(self, name: str) -> str:
        return os.path.join(self.settings.data_dir, "vectorstore", name)

    def answer_question(self, kb: str, question: str, top_k: int = 4) -> tuple[str, List[dict]]:
        """
        执行RAG问答检索
        
        检索流程详解：
        1. 创建向量检索器：从指定知识库的向量存储中加载数据
        2. 向量相似度搜索：将用户问题向量化，在向量空间中搜索最相似的文档
        3. 构建上下文：将检索到的文档组织成格式化的上下文
        4. 生成回答：将上下文和问题一起发送给LLM生成回答
        
        Args:
            kb: 知识库名称
            question: 用户问题
            top_k: 返回最相似的文档数量（默认4个）
            
        Returns:
            元组：(回答内容, 相关文档列表)
        """
        # 步骤1：创建向量检索器，指定知识库和返回数量
        retriever = self.vs.as_retriever(self._kb_vector_dir(kb), top_k)
        
        # 步骤2：执行向量相似度搜索
        # 原理：将问题文本向量化，计算与存储向量的相似度，返回最相似的文档
        # 实现：get_relevant_documents() 是 LangChain 框架提供的方法
        # 底层调用：Chroma向量数据库的相似度搜索功能
        docs = retriever.get_relevant_documents(question)
        
        # 步骤3：构建格式化的上下文，包含文档来源和内容
        # 详细说明：将检索到的文档格式化为标准上下文格式
        # - 每个文档用序号标记 [1], [2], [3]...
        # - 显示文档来源路径，便于引用和验证
        # - 包含完整的文档内容，为LLM提供充分的背景信息
        # - 用双换行符分隔不同文档，提高可读性
        context = "\n\n".join([f"[{i+1}] ({d.metadata.get('source','')})\n{d.page_content}" for i, d in enumerate(docs)])

        # 步骤4：构建消息并生成回答
        # 消息结构说明：
        # SystemMessage: 设置AI助手的系统级行为指导
        # - 要求使用提供的上下文回答问题
        # - 要求引用来源路径
        # - 要求诚实回答不知道的情况
        #
        # HumanMessage: 包含具体的上下文和用户问题
        # - 首先提供检索到的相关文档作为上下文
        # - 然后给出用户的具体问题
        # - 让LLM基于这些上下文信息生成准确回答
        messages = [
            SystemMessage(content=SYS_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]

        # 调用LLM生成回答
        # 处理可能的响应格式差异，确保获取文本内容
        resp = self._llm().invoke(messages)
        answer = resp.content if hasattr(resp, "content") else str(resp)
        
        # 整理来源信息，用于展示引用
        sources = [{"source": d.metadata.get("source", ""), "snippet": d.page_content[:300]} for d in docs]
        return answer, sources


