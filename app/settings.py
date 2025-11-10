from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 嵌入模型配置（用于文本向量化）
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "deepseek-r1:1.5b")
    
    # 问答模型配置（用于对话生成）
    chat_model: str = os.getenv("CHAT_MODEL", "deepseek-r1:1.5b")
    
    data_dir: str = os.getenv("DATA_DIR", "./data")

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # Ensure base folders exist (with error handling for permission issues)
    # 在 Docker 环境中，/data 目录是挂载卷，可能没有写入权限
    # 改为使用应用工作目录下的 data 目录
    try:
        # 使用相对路径，避免权限问题
        data_dir = "./data"  # 使用相对路径，在 /app 目录下
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "vectorstore"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "kb"), exist_ok=True)
        
        # 更新设置中的 data_dir 为相对路径
        settings.data_dir = data_dir
        
    except PermissionError:
        # Log the error but don't crash the application
        import logging
        logging.warning(f"Permission denied when creating directories. "
                      f"Application will continue but some features may not work properly.")
    return settings


