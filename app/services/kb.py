import os
import shutil
from typing import List, Iterable

from fastapi import UploadFile

from ..settings import Settings
from .vectorstore import VectorStore
from ..utils.chunking import chunk_texts


# 支持的文件扩展名集合，用于验证上传文件类型
SUPPORTED_EXTS = {
    ".txt", ".md", ".py", ".java", ".sql", ".json", ".csv"
}


class KnowledgeBaseService:
    """
    知识库服务类，负责管理知识库的创建、删除、文件存储和向量索引构建
    
    该服务提供了通过文件上传和Git仓库导入两种方式构建知识库的能力，
    并将处理后的文档内容存储到向量数据库中，以便后续检索。
    """
    
    def __init__(self, settings: Settings) -> None:
        """
        初始化知识库服务
        
        Args:
            settings: 应用程序设置对象，包含数据存储路径等配置信息
        """
        self.settings = settings

    # ------------------------------
    # 路径管理方法
    # ------------------------------
    
    def kb_root(self, name: str) -> str:
        """
        获取知识库的根目录路径
        
        Args:
            name: 知识库名称
            
        Returns:
            知识库根目录的绝对路径
        """
        return os.path.join(self.settings.data_dir, "kb", name)

    def kb_source_dir(self, name: str) -> str:
        """
        获取知识库的源文件存储目录路径
        
        Args:
            name: 知识库名称
            
        Returns:
            源文件存储目录的绝对路径
        """
        return os.path.join(self.kb_root(name), "source")

    def kb_vector_dir(self, name: str) -> str:
        """
        获取知识库的向量存储目录路径
        
        Args:
            name: 知识库名称
            
        Returns:
            向量存储目录的绝对路径
        """
        return os.path.join(self.settings.data_dir, "vectorstore", name)

    # ------------------------------
    # 知识库管理方法
    # ------------------------------
    
    def create_kb(self, name: str) -> None:
        """
        创建新的知识库目录结构
        
        该方法会创建知识库的根目录、源文件目录和向量存储目录，
        如果目录已存在则不会抛出异常。
        
        Args:
            name: 知识库名称
        """
        os.makedirs(self.kb_source_dir(name), exist_ok=True)
        os.makedirs(self.kb_vector_dir(name), exist_ok=True)

    def list_kb(self) -> List[str]:
        """
        列出所有可用的知识库名称
        
        Returns:
            按字母排序的知识库名称列表，如果没有知识库则返回空列表
        """
        base = os.path.join(self.settings.data_dir, "kb")
        if not os.path.exists(base):
            return []
        return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    def delete_kb(self, name: str) -> None:
        """
        删除指定的知识库及其所有内容
        
        该方法会删除知识库的源文件目录和向量存储目录，
        如果目录不存在则忽略错误。
        
        Args:
            name: 要删除的知识库名称
        """
        shutil.rmtree(self.kb_root(name), ignore_errors=True)
        shutil.rmtree(self.kb_vector_dir(name), ignore_errors=True)

    # ------------------------------
    # 文档导入方法
    # ------------------------------
    
    async def save_and_ingest_files(self, name: str, files: List[UploadFile]) -> int:
        """
        保存上传的文件并将其内容导入知识库
        
        处理流程:
        1. 创建知识库目录结构
        2. 验证并保存上传的文件
        3. 读取文件内容并分块处理
        4. 将分块后的文本添加到向量存储
        
        Args:
            name: 目标知识库名称
            files: 从FastAPI接收到的上传文件列表
            
        Returns:
            成功导入向量存储的文档块数量
        """
        self.create_kb(name)
        saved_paths: List[str] = []
        
        # 保存上传的文件
        for f in files:
            # 获取文件扩展名并转换为小写
            ext = os.path.splitext(f.filename or "")[1].lower()
            if ext not in SUPPORTED_EXTS:
                # 跳过不支持的文件类型，不抛出错误
                continue
                
            # 构建目标文件路径
            target = os.path.join(self.kb_source_dir(name), f.filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            
            # 保存文件内容
            with open(target, "wb") as out:
                out.write(await f.read())
            saved_paths.append(target)

        # 处理文档并添加到向量存储
        docs = self._collect_docs(saved_paths)
        VectorStore(self.settings).add_documents(self.kb_vector_dir(name), docs)
        return len(docs)

    def ingest_git_repo(self, name: str, repo_url: str, branch: str | None, username: str | None, token: str | None) -> int:
        """
        从Git仓库导入代码文件并构建知识库
        
        处理流程:
        1. 创建知识库目录结构
        2. 克隆Git仓库到临时目录
        3. 筛选并复制支持的文件类型到知识库
        4. 读取文件内容并分块处理
        5. 将分块后的文本添加到向量存储
        
        Args:
            name: 目标知识库名称
            repo_url: Git仓库URL
            branch: 要克隆的分支名称，None表示使用默认分支
            username: Git认证用户名，None表示不需要认证
            token: Git认证令牌，None表示不需要认证
            
        Returns:
            成功导入向量存储的文档块数量
        """
        from git import Repo
        self.create_kb(name)
        
        # 准备临时目录
        tmp_dir = os.path.join(self.kb_root(name), "git_tmp")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        # 构建带有认证信息的URL（如果提供）
        url = repo_url
        if username and token and repo_url.startswith("https://") and "@" not in repo_url:
            url = repo_url.replace("https://", f"https://{username}:{token}@")

        # 克隆Git仓库
        Repo.clone_from(url, tmp_dir, branch=branch or None)

        # 复制支持的文件到知识库源目录
        saved_paths: List[str] = []
        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                    
                src = os.path.join(root, fn)
                # 计算相对路径，保持仓库内的目录结构
                rel = os.path.relpath(src, tmp_dir)
                dst = os.path.join(self.kb_source_dir(name), rel)
                
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                saved_paths.append(dst)

        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # 处理文档并添加到向量存储
        docs = self._collect_docs(saved_paths)
        VectorStore(self.settings).add_documents(self.kb_vector_dir(name), docs)
        return len(docs)

    # ------------------------------
    # 内部辅助方法
    # ------------------------------
    
    def _collect_docs(self, paths: List[str]):
        """
        读取文件内容并分块处理，为向量存储准备文档数据
        
        Args:
            paths: 要处理的文件路径列表
            
        Returns:
            包含文本块和元数据的元组列表，格式为[(text_chunk, metadata), ...]
        """
        texts = []
        metadatas = []
        
        for p in paths:
            try:
                # 读取文件内容，使用utf-8编码并忽略解码错误
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    
                # 将文本分块并添加元数据
                for chunk in chunk_texts(content):
                    texts.append(chunk)
                    metadatas.append({"source": p})
                    
            except Exception:
                # 忽略处理失败的文件
                continue
                
        return list(zip(texts, metadatas))
