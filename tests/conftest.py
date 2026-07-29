"""
pytest 共享配置

conftest.py 是 pytest 的特殊文件——它里面的 fixture（测试工具）
可以被 tests/ 下所有的测试文件自动使用，不需要 import。

作用：
  1. 提供统一的测试配置（Settings），避免每个测试文件都重复创建
  2. 提供共享的测试工具（mock 的 embedding/client 等）
  3. fixture 的 scope 控制资源生命周期（如临时目录自动清理）
"""

import tempfile
from typing import Generator

import pytest
from pytest import FixtureRequest

from app.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    """
    创建一个测试专用的 Settings 实例。

    为什么不用全局的 get_settings()？
    ─────────────────────────────────────
    get_settings() 用了 @lru_cache，在同一个进程里只会创建一次。
    测试时如果需要不同的配置（比如不同 data_dir），就要创建新实例。
    这里每次测试传一个临时目录，避免测试数据污染。

    为什么要用临时目录？
    ─────────────────────────────────────
    data_dir 默认是 ./data，如果测试往里面写东西，会污染开发数据。
    用 tempfile.mkdtemp() 每次测试创建临时目录，测试完自动丢弃。
    """
    return Settings(
        ollama_base_url="http://test:11434",
        embedding_model="test-embed-model",
        chat_model="test-chat-model",
        data_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def kb_vector_dir(settings: Settings) -> str:
    """
    测试用的向量存储目录（基于 settings 的临时 data_dir）。

    每个测试有自己的临时目录，互不干扰。
    """
    import os
    return os.path.join(settings.data_dir, "vectorstore", "test_kb")


@pytest.fixture
def vector_store(settings: Settings):
    """
    创建一个 VectorStore 实例用于测试。

    注入的是测试用的 settings（指向临时目录、mock 地址），
    不会影响真实的 ChromaDB 数据。
    """
    from app.infrastructure.vector_store import VectorStore
    return VectorStore(settings)
