from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 嵌入模型配置（用于文本向量化）
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    
    # 问答模型配置（用于对话生成）
    chat_model: str = os.getenv("CHAT_MODEL", "qwen3.5:2b")
    
    data_dir: str = os.getenv("DATA_DIR", "./data")

    # LLM 运行参数
    # 思考模式：思考型模型（如 qwen3）是否输出 think 阶段
    llm_think: bool = True if os.getenv("LLM_THINK", "true").lower() == "true" else False
    # 单次生成的最大 token 数；None 或空值表示不限制
    llm_max_tokens: Optional[int] = None
    # 上下文窗口大小（token）：Agent 的消息历史（system + 工具调用 + 结果）
    llm_num_ctx: int = int(os.getenv("LLM_NUM_CTX", "8192"))

    @field_validator("llm_max_tokens", mode="before")
    @classmethod
    def empty_max_tokens_to_none(cls, v):
        """允许 .env 中留空（''）或写 'None'，统一转为 None"""
        if v is None:
            return None
        if isinstance(v, str) and (v.strip() == "" or v.strip().lower() == "none"):
            return None
        return v

    # 网络超时配置（单位：秒）
    git_timeout: int = int(os.getenv("GIT_TIMEOUT", "300"))  # 默认5分钟
    git_connect_timeout: int = int(os.getenv("GIT_CONNECT_TIMEOUT", "30"))  # 默认30秒

    # Git 加速器配置
    # 开关：是否启用 GitHub 镜像加速
    git_accelerator_enabled: bool = os.getenv("GIT_ACCELERATOR_ENABLED", "true").lower() == "true"
    # 加速镜像列表，逗号分隔，按优先级从前到后尝试。
    # 支持两种模式：
    #   prefix|https://xxx    前缀代理：https://xxx/https://github.com/user/repo
    #                         （适合 ghproxy 类、akams.cn 类透传代理）
    #   replace|https://xxx   域名替换：https://github.com/user/repo
    #                         → https://xxx/user/repo，并映射 raw/api/codeload 子域名
    git_accelerators: str = os.getenv(
        "GIT_ACCELERATORS",
        "prefix|https://github.akams.cn,"
        "replace|https://bgithub.xyz",
    )

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


