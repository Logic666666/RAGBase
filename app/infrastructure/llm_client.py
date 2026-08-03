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

import re
from dataclasses import dataclass

import httpx

# 匹配 <think>...</think> 标签（非贪婪，支持多行）
_THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_think_tags(text: str) -> tuple[str, str]:
    """
    从文本中提取 <think>...</think> 标签内容，并剥离标签返回正文。

    兼容处理：部分 Ollama 版本或 OpenAI 兼容服务器
    不提供结构化的 message.thinking 字段，
    而是把思考内容以 <think> 标签形式混在 content 里。

    Returns:
        (thinking, content_without_tags)
    """
    matches = _THINK_TAG_PATTERN.findall(text)
    if not matches:
        return "", text

    thinking = "\n".join(m.strip() for m in matches if m.strip())
    content = _THINK_TAG_PATTERN.sub("", text).strip()
    return thinking, content


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

    - content:   LLM 回复的正文文本
    - thinking:  模型的推理过程（思考内容）。
                 来自 Ollama 的结构化 message.thinking 字段，
                 或从 content 中的 <think>...</think> 标签提取。
                 无思考时为空字符串。
    - done:      是否完成（非流式永远 True，流式预留）
    """
    content: str
    thinking: str = ""
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
        think: bool = False,
        max_tokens: int | None = None,
    ):
        """
        Args:
            base_url:    Ollama 服务地址（如 "http://localhost:11434"）
            model:       对话模型名称（如 "qwen3:4b"）
            temperature: 生成温度（0~1，越低越确定，越高越随机）
            timeout:     HTTP 请求超时秒数。
            think:       是否启用思考型模型的 think 阶段。
                         关闭可大幅提速（Agent 已有显式 thought）。
            max_tokens:  单次生成的最大 token 数；None 表示不限制。
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.think = think
        self.max_tokens = max_tokens


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
        # 动态组装 payload：
        # think / max_tokens 由配置注入（.env 可配），
        # max_tokens 为 None 时不传（不限制长度）
        options: dict = {
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens

        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "stream": False,
            "think": self.think,
            "options": options,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Ollama 返回格式：
        # {"message": {"role": "assistant", "thinking": "...", "content": "..."}, "done": true}
        # 思考型模型（如 qwen3）在 message.thinking 字段返回推理过程
        message = data.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "") or ""
        done = data.get("done", True)

        # 双通道提取思考内容：
        # 通道 1：message.thinking 结构化字段（新版 Ollama）
        # 通道 2：content 中的 <think>...</think> 标签（旧版 Ollama / 兼容服务器）
        # 兼容处理：若结构化字段缺失，从 content 提取标签并剥离
        if not thinking:
            thinking, content = _extract_think_tags(content)

        return ChatResponse(content=content, thinking=thinking, done=done)
