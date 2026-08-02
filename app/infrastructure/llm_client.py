"""
大语言模型（LLM）对话客户端模块

直接通过 HTTP API 调用 Ollama 的 /api/chat 接口，
不依赖 LangChain 的 ChatOllama。

使用示例：
    client = OllamaChatClient("http://localhost:11434", "deepseek-r1:1.5b")
    answer = await client.chat([
        Message(role="system", content="你是助手"),
        Message(role="user", content="你好"),
    ])
    # → "你好！我是助手。"
"""

from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class Message:
    """
    对话消息。

    role 的可选值及其含义：
    ─────────────────────────────────────────────
    "system"    系统指令——设定 LLM 的行为和风格
    "user"      用户输入
    "assistant" LLM 的回复（在历史对话中出现）
    "tool"      工具执行结果
    ─────────────────────────────────────────────
    """
    role: str
    content: str


@dataclass
class ChatResponse:
    """
    LLM 对话的完整响应。

    不再像 LangChain 的 AIMessage 那样封装一堆内部状态，
    只保留需要的信息：
    - content:  LLM 回复的文本
    - done:    是否完成（非流式永远 True，流式预留）
    """
    content: str
    done: bool = True


class OllamaChatClient:
    """
    Ollama 对话客户端。

    直接调 Ollama 的 /api/chat 接口：
    POST /api/chat
    {
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": false,
        "options": {"temperature": 0.2}
    }

    不使用 LangChain 的 ChatOllama：
    1. 理解 LLM 调用的完整协议而不仅仅是调一个方法
    2. 控制参数（temperature、max_tokens 等）——LangChain 的封装会干扰参数传递
    3. 为后续 tool calling 和 streaming 铺路
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout: int = 300,
    ):
        """
        Args:
            base_url:    Ollama 服务地址（如 "http://localhost:11434"）
            model:       对话模型名称（如 "qwen3:4b"）
            temperature: 生成温度（0~1，越低越确定，越高越随机）
            timeout:     HTTP 请求超时秒数。
                         CPU 推理大模型较慢（尤其是思考型模型），
                         5 分钟（300 秒）较为充裕
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout


    # 公开接口
    async def chat(self, messages: list[Message]) -> str:
        """
        发送对话消息，获取 LLM 回复。

        agent 循环中核心调用：
        - LLM 回复可能是简单的问→答
        - 可能是"我想调用一个工具"，
          此时这个接口会返回 ChatResponse 而不是 str，
          包含 tool_calls 字段

        参数:
            messages: 对话历史（system prompt + user 输入 + 历史往返）

        返回:
            LLM 回复文本

        异常:
            httpx.TimeoutException:  请求超时
            httpx.HTTPStatusError:   Ollama 返回非 2xx
            httpx.RequestError:      网络错误
        """
        response = await self._request(messages)
        return response.content

    async def chat_with_response(self, messages: list[Message]) -> ChatResponse:
        """
        和 chat() 一样，但返回完整的 ChatResponse 对象。
        引入 tool calling 后，ChatResponse 会包含 tool_calls 字段。
        现在先预留接口。
        """
        return await self._request(messages)


    # 内部方法
    async def _request(self, messages: list[Message]) -> ChatResponse:
        """
        发送请求到 Ollama /api/chat。

        Ollama API 格式：
            POST /api/chat
            {
                "model": "模型名",
                "messages": [{"role": "user", "content": "你好"}],
                "stream": false,
                "options": {
                    "temperature": 0.2
                }
            }

            响应：
            {
                "model": "deepseek-r1:1.5b",
                "message": {"role": "assistant", "content": "你好！"},
                "done": true
            }
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Ollama 返回格式：{"message": {"role": "assistant", "content": "..."}, "done": true}
        message = data.get("message", {})
        content = message.get("content", "")
        done = data.get("done", True)

        return ChatResponse(content=content, done=done)
