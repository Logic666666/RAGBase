"""
项目冒烟测试

验证最基本的项目状态：
  1. 没有 LangChain 残留
  2. 所有模块能正常 import
  3. 核心类接口正确

冒烟测试（smoke test）的意思是：
"系统能不能启动？"——不验证功能细节，只验证"没坏到不能用的程度"。
"""


def test_no_langchain_in_environment():
    """
    验证环境中没有安装 langchain 相关包。

    这是 Phase 1（脱框）的核心目标：
    不再依赖 LangChain 的任何组件。
    """
    import importlib
    langchain_packages = [
        "langchain", "langchain_core", "langchain_community",
        "langchain_ollama", "langchain_text_splitters",
    ]
    for pkg in langchain_packages:
        spec = importlib.util.find_spec(pkg)
        assert spec is None, (
            f"⚠️ {pkg} 仍然存在于环境中！"
            f"Phase 1 要求移除所有 LangChain 依赖。"
        )


def test_infrastructure_modules_import():
    """
    验证基础设施层所有模块能正常 import。

    这是保证系统"可启动"的最小条件。
    """
    from app.infrastructure import embeddings         # noqa
    from app.infrastructure import llm_client         # noqa
    from app.infrastructure import text_splitter      # noqa
    from app.infrastructure import vector_store       # noqa


def test_infrastructure_classes_instantiate():
    """
    验证核心类能被实例化。

    （不需要真实连接，只需要构造成功）
    """
    from app.infrastructure.embeddings import OllamaEmbeddingClient
    from app.infrastructure.llm_client import OllamaChatClient

    # 只需要构造，不需要调方法
    _ = OllamaEmbeddingClient(base_url="http://test:11434", model="test")
    _ = OllamaChatClient(base_url="http://test:11434", model="test")


def test_text_splitter_accessible():
    """
    text_splitter 的核心函数 split_text 可以被调用。
    kb.py 依赖它，所以这是必须保证的接口稳定性。
    """
    from app.infrastructure.text_splitter import split_text
    # 不测试功能（功能有专门的单元测试），只测试"能调"
    result = split_text("test")
    assert isinstance(result, list)
