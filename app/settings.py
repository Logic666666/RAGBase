from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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


