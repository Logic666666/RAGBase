from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import uuid
import logging
from loguru import logger

from models.config import settings
from services.document_processor import DocumentProcessor
from services.git_processor import GitProcessor
from services.vector_store import VectorStoreService
from services.ollama_service import OllamaService
from schemas.request import (
    QueryRequest, 
    GitRepositoryRequest,
    KnowledgeBaseRequest
)
from schemas.response import (
    QueryResponse,
    DocumentProcessingResponse,
    GitProcessingResponse,
    HealthCheckResponse
)

# 创建 API 路由
router = APIRouter(
    tags=["RAG Knowledge Base API"],
    responses={404: {"description": "Not found"}},
)

# 初始化服务
document_processor = DocumentProcessor()
git_processor = GitProcessor()
vector_store = VectorStoreService()
ollama_service = OllamaService()

# 获取知识库列表接口
@router.get("/kb")
async def list_knowledge_bases():
    """获取所有知识库列表"""
    try:
        from services.kb import KnowledgeBaseService
        kb_service = KnowledgeBaseService(settings)
        kb_list = kb_service.list_kb()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "knowledge_bases": kb_list
            }
        )
    except Exception as e:
        logger.error(f"获取知识库列表失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"获取知识库列表失败: {str(e)}"
            }
        )

# 健康检查接口
@router.get("/health", response_model=HealthCheckResponse)
async def api_health_check():
    """API 健康检查接口"""
    return {
        "status": "healthy",
        "service": "RAG Knowledge Base API",
        "version": "1.0.0"
    }

# 文档上传接口
@router.post("/documents/upload", response_model=DocumentProcessingResponse)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_name: str = Form(...)
):
    """上传文档并添加到知识库"""
    try:
        # 生成唯一文件名
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(settings.upload_dir, unique_filename)
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # 处理文档
        logger.info(f"开始处理文档: {file.filename}")
        document_id, chunks = document_processor.process_document(
            file_path=file_path,
            knowledge_base_name=knowledge_base_name,
            file_name=file.filename
        )
        
        # 向量化并存储
        vector_store.add_documents(
            chunks=chunks,
            knowledge_base_name=knowledge_base_name
        )
        
        # 删除临时文件
        os.remove(file_path)
        
        logger.info(f"文档处理完成: {file.filename}, 生成 {len(chunks)} 个片段")
        return {
            "status": "success",
            "document_id": document_id,
            "file_name": file.filename,
            "chunks_count": len(chunks),
            "knowledge_base": knowledge_base_name
        }
    except Exception as e:
        logger.error(f"文档处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

# 知识库删除接口
@router.delete("/kb/{knowledge_base_name}")
async def delete_knowledge_base(knowledge_base_name: str):
    """删除指定知识库"""
    try:
        from services.kb import KnowledgeBaseService
        kb_service = KnowledgeBaseService(settings)
        
        # 调用删除方法并检查结果
        deletion_success = kb_service.delete_kb(knowledge_base_name)
        
        if deletion_success:
            return JSONResponse(
                status_code=200,
                content={"status": "success", "message": f"知识库 '{knowledge_base_name}' 已成功删除"}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"删除知识库 '{knowledge_base_name}' 失败，请检查权限或文件是否被占用"}
            )
    except Exception as e:
        logger.error(f"删除知识库时发生错误: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"删除知识库失败: {str(e)}"}
        )

# Git 仓库解析接口
@router.post("/git/parse", response_model=GitProcessingResponse)
async def parse_git_repository(request: GitRepositoryRequest):
    """解析 Git 仓库并添加到知识库"""
    try:
        logger.info(f"开始解析 Git 仓库: {request.repository_url}")
        
        # 克隆仓库
        repo_path = git_processor.clone_repository(
            repo_url=request.repository_url,
            username=request.username,
            password=request.password,
            branch=request.branch
        )
        
        # 处理仓库文件
        document_id, chunks = git_processor.process_repository(
            repo_path=repo_path,
            knowledge_base_name=request.knowledge_base_name,
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns
        )
        
        # 向量化并存储
        vector_store.add_documents(
            chunks=chunks,
            knowledge_base_name=request.knowledge_base_name
        )
        
        # 清理临时目录
        git_processor.cleanup(repo_path)
        
        logger.info(f"Git 仓库解析完成: {request.repository_url}, 生成 {len(chunks)} 个片段")
        return {
            "status": "success",
            "repository_url": request.repository_url,
            "branch": request.branch,
            "document_id": document_id,
            "chunks_count": len(chunks),
            "knowledge_base": request.knowledge_base_name
        }
    except Exception as e:
        logger.error(f"Git 仓库解析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Git 仓库解析失败: {str(e)}")

# 问答接口
@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """查询知识库"""
    try:
        logger.info(f"收到查询请求: {request.query} (知识库: {request.knowledge_base_name})")
        
        # 检索相关文档
        relevant_docs = vector_store.similarity_search(
            query=request.query,
            knowledge_base_name=request.knowledge_base_name,
            top_k=request.top_k or 5
        )
        
        # 构建提示
        prompt = ollama_service.build_rag_prompt(
            query=request.query,
            context_docs=relevant_docs
        )
        
        # 获取模型响应
        response = ollama_service.generate_response(
            prompt=prompt,
            model=request.model or settings.ollama_model
        )
        
        logger.info(f"查询处理完成: {request.query}")
        return {
            "status": "success",
            "query": request.query,
            "response": response,
            "knowledge_base": request.knowledge_base_name,
            "sources": [doc.metadata for doc in relevant_docs]
        }
    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

# 获取知识库列表接口
@router.get("/knowledge-bases")
async def list_knowledge_bases():
    """获取所有知识库列表"""
    try:
        knowledge_bases = vector_store.list_collections()
        return {
            "status": "success",
            "knowledge_bases": knowledge_bases
        }
    except Exception as e:
        logger.error(f"获取知识库列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取知识库列表失败: {str(e)}")

# 删除知识库接口
@router.delete("/knowledge-bases/{knowledge_base_name}")
async def delete_knowledge_base(knowledge_base_name: str):
    """删除指定知识库"""
    try:
        vector_store.delete_collection(knowledge_base_name)
        return {
            "status": "success",
            "message": f"知识库 '{knowledge_base_name}' 已成功删除",
            "knowledge_base": knowledge_base_name
        }
    except Exception as e:
        logger.error(f"删除知识库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除知识库失败: {str(e)}")