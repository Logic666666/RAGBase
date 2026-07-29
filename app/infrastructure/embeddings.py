"""
文本嵌入客户端模块

将文本转换为向量表示（embedding），用于语义检索。
不依赖 LangChain 的 OllamaEmbeddings，而是自己实现一个轻量客户端。
直接通过 HTTP API 调用 Ollama 的 /api/embeddings 接口。

使用示例：
    client = OllamaEmbeddingClient("http://localhost:11434", "bge-m3")
    vec = await client.embed_query("你好世界")
    # → [0.012, 0.345, ..., 0.789]  共 768 维
"""

from typing import Optional

import httpx


class OllamaEmbeddingClient:
    """
    Ollama 嵌入客户端。

    设计思路：
    embedding 把自然语言翻译成向量语言。
    这个向量语言的质量直接决定了检索的好坏。

    我们选择在"客户端"做 embedding（自己调 API 算好向量再传给向量库），
    而不是让向量库自动调 embedding function。
    原因：
    1. 清楚每一步在干什么（不是黑盒）
    2. 可以加缓存、加日志、加错误处理
    3. 将来换向量库（Chroma → Milvus）不需要换 embedding 方式
    4. 方便测试——可以 mock 这层
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 60,
    ):
        """
        Args:
            base_url: Ollama 服务地址（如 "http://localhost:11434"）
            model:    嵌入模型名称（如 "bge-m3"、"nomic-embed-text"）
            timeout:  HTTP 请求超时秒数（embedding 模型通常很快，但大文档可能慢）
        """
        # 去掉尾部斜杠，方便拼接路径
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout


    # 公开接口：单条 embedding
    async def embed_query(self, text: str) -> list[float]:
        """
        将一段文本转换为向量。

        1. 把文本发到 Ollama
        2. Ollama 用指定的模型算出一个向量
        3. 返回这个向量

        Args:
            text: 要向量化的文本

        Returns:
            浮点数列表（向量的维度取决于模型，如 bge-m3 是 1024 维）

        Raises:
            httpx.HTTPError: 网络错误或 Ollama 无响应
        """
        result = await self._request(text)

        # Ollama 返回格式：{"embedding": [0.1, 0.2, ...], "prompt": "原始文本"}
        embedding = result.get("embedding")
        if embedding is None:
            raise ValueError(
                f"Ollama 返回了意外的响应格式（缺少 'embedding' 字段）: {result}"
            )

        return embedding


    # 公开接口：批量 embedding
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量将多段文本转换为向量。

        目前采用逐个调用的方式（循环 embed_query）。
        如果未来需要高性能批量处理，可以改为使用 Ollama 的 /api/embed 接口。

        很多 embedding 模型内部有长度限制（如 512 tokens），
        分批处理可以更好地控制每段的长度，避免截断。
        """
        results: list[list[float]] = []
        for text in texts:
            vec = await self.embed_query(text)
            results.append(vec)
        return results

    # ──────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────

    async def _request(self, text: str) -> dict:
        """
        发送 HTTP 请求到 Ollama 的 /api/embeddings 接口。

        Ollama API 格式：
            POST /api/embeddings
            {
                "model": "模型名",
                "prompt": "要向量化的文本"
            }
            响应:
            {
                "embedding": [0.012, 0.345, ...],
                "prompt": "原始文本"
            }

        Raises:
            httpx.TimeoutException:  请求超时（通常网络问题或 Ollama 负载过高）
            httpx.HTTPStatusError:   非 200 响应（如 Ollama 返回 404/500）
            httpx.RequestError:      其他网络错误
        """
        payload = {
            "model": self.model,
            "prompt": text,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
            )
            # 非 2xx 响应会抛出 HTTPStatusError
            resp.raise_for_status()
            return resp.json()
