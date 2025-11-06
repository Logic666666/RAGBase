import os
import logging
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic_settings import BaseSettings
from loguru import logger
import uvicorn

# 确保必要的目录存在
os.makedirs(os.getenv("UPLOAD_DIR", "./uploads"), exist_ok=True)
os.makedirs(os.getenv("GIT_CLONE_DIR", "./git_repos"), exist_ok=True)
os.makedirs(os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db"), exist_ok=True)

# 配置日志
logger.add("app.log", rotation="500 MB", level="INFO")
logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn").handlers = []

# 加载环境变量配置
class Settings(BaseSettings):
    app_name: str = os.getenv("APP_NAME", "DeepSeek RAG Knowledge Base")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Ollama 配置
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "deepseek-embedding")
    
    # 向量数据库配置
    vector_store: str = os.getenv("VECTOR_STORE", "chroma")
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db")
    
    # 文件上传配置
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    
    # Git 配置
    git_clone_dir: str = os.getenv("GIT_CLONE_DIR", "./git_repos")

# 初始化配置
settings = Settings()

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.app_name,
    description="基于 Ollama 和 DeepSeek 的 RAG 知识库系统",
    version="1.0.0",
    debug=settings.debug
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 根路由 - 渲染主页面
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 健康检查路由
@app.get("/health")
async def health_check():
    return {"status": "healthy", "app_name": settings.app_name}

# 导入并注册其他路由
from api import router as api_router
app.include_router(api_router, prefix="/api")

# 应用入口
if __name__ == "__main__":
    logger.info(f"Starting {settings.app_name} on {settings.app_host}:{settings.app_port}")
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info"
    )