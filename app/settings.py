from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 嵌入模型配置（用于文本向量化）
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    # 问答模型配置（用于对话生成）
    chat_model: str = os.getenv("CHAT_MODEL", "deepseek-r1:1.5b")
    
    # 为了向后兼容，保留旧的配置项
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-r1:1.5b")
    
    data_dir: str = os.getenv("DATA_DIR", "./data")

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # Ensure base folders exist
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(os.path.join(settings.data_dir, "vectorstore"), exist_ok=True)
    os.makedirs(os.path.join(settings.data_dir, "kb"), exist_ok=True)
    return settings


