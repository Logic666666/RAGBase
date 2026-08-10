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

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ..reliability.retry import RetryPolicy

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
    - done_reason: 结束原因（对齐 Claude Code 的 stop_reason）：
                 "stop"         正常完成
                 "length"       达到 num_predict 上限（输出被截断）
                 "model_length" 达到模型上下文上限
                 记录它用于诊断"输出截断"类问题，不再靠猜。
    """
    content: str
    thinking: str = ""
    done: bool = True
    done_reason: str = "stop"


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
        num_ctx: int = 8192,
        retry_policy: "RetryPolicy | None" = None,
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
            num_ctx:     上下文窗口大小（token）。Agent 的消息历史
                         （system + 工具调用 + 结果）占用大量上下文，
                         Ollama 默认 2048/4096 会导致回答被 length 截断。
            retry_policy: 重试策略（transient 错误自动退避重试）。
                          默认 ExponentialBackoff（对齐 Claude Code 的
                          retry with backoff，单轮上限 3 次）。
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.think = think
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        from ..reliability.retry import ExponentialBackoff
        self.retry_policy = retry_policy or ExponentialBackoff()


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
        # think / max_tokens / num_ctx 由配置注入（.env 可配），
        # max_tokens 为 None 时不传（不限制长度）
        options: dict = {
            "temperature": self.temperature,
            # 上下文窗口：Agent 消息历史占用大，默认 2048/4096 会导致
            # 回答被 done_reason=length 截断
            "num_ctx": self.num_ctx,
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

        # 带重试的请求（对齐 Claude Code 的 retry with backoff）：
        # transient 错误（网络闪断/超时/5xx）自动退避重试，
        # permanent 错误直接抛出——重试只会重复失败
        attempt = 0
        while True:
            try:
                data = await self._send_request(payload)
                break
            except Exception as e:
                from ..reliability.errors import classify_error
                category = classify_error(e)
                if not self.retry_policy.should_retry(attempt, category):
                    raise
                delay = self.retry_policy.next_delay(attempt)
                await asyncio.sleep(delay)
                attempt += 1

        # Ollama 返回格式：
        # {"message": {"role": "assistant", "thinking": "...", "content": "..."}, "done": true}
        # 思考型模型（如 qwen3）在 message.thinking 字段返回推理过程
        message = data.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "") or ""
        done = data.get("done", True)
        # 结束原因（诊断截断的关键）："stop"正常 / "length"触顶 / "model_length"上下文满
        done_reason = data.get("done_reason", "stop")

        # 双通道提取思考内容：
        # 通道 1：message.thinking 结构化字段（新版 Ollama）
        # 通道 2：content 中的 <think>...</think> 标签（旧版 Ollama / 兼容服务器）
        # 兼容处理：若结构化字段缺失，从 content 提取标签并剥离
        if not thinking:
            thinking, content = _extract_think_tags(content)

        return ChatResponse(content=content, thinking=thinking, done=done,
                            done_reason=done_reason)

    async def _send_request(self, payload: dict) -> dict:
        """
        发送一次 HTTP 请求并返回响应 JSON。

        单独抽出来便于重试循环调用（每次重试都是全新请求）。
        异常（超时/连接/HTTP 状态）由调用方的重试循环处理。
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
