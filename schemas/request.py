from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询问题")
    knowledge_base_name: str = Field(..., description="知识库名称")
    top_k: Optional[int] = Field(5, description="检索相关文档的数量")
    model: Optional[str] = Field(None, description="使用的模型名称")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        None, description="聊天历史记录"
    )

class GitRepositoryRequest(BaseModel):
    """Git 仓库解析请求模型"""
    repository_url: HttpUrl = Field(..., description="Git 仓库 URL")
    knowledge_base_name: str = Field(..., description="目标知识库名称")
    username: Optional[str] = Field(None, description="Git 仓库用户名")
    password: Optional[str] = Field(None, description="Git 仓库密码或访问令牌")
    branch: Optional[str] = Field("main", description="要克隆的分支名称")
    include_patterns: Optional[List[str]] = Field(
        None, description="要包含的文件模式列表"
    )
    exclude_patterns: Optional[List[str]] = Field(
        None, description="要排除的文件模式列表"
    )

class KnowledgeBaseRequest(BaseModel):
    """知识库操作请求模型"""
    knowledge_base_name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")

class DocumentProcessingRequest(BaseModel):
    """文档处理请求模型"""
    file_path: str = Field(..., description="文档文件路径")
    knowledge_base_name: str = Field(..., description="目标知识库名称")
    chunk_size: Optional[int] = Field(1000, description="文档分块大小")
    chunk_overlap: Optional[int] = Field(200, description="分块重叠大小")