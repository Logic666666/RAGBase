"""
可靠性层单元测试

验证：
  1. 错误分类（transient / permanent）
  2. 重试策略（次数限制、退避计算）
  3. llm_client 的重试行为（transient 自动重试、permanent 直接抛）
"""

from unittest.mock import patch

import httpx
import pytest

from app.reliability import ErrorCategory, ExponentialBackoff, classify_error


# ──────────────────────────────────────────────
# 错误分类
# ──────────────────────────────────────────────

def test_classify_connect_error_transient():
    """连接失败 → transient（可重试）"""
    exc = httpx.ConnectError("connection refused")
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_classify_timeout_transient():
    """超时 → transient"""
    exc = httpx.ReadTimeout("read timeout")
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_classify_429_transient():
    """限流 429 → transient"""
    exc = httpx.HTTPStatusError("429", request=httpx.Request("POST", "http://x"),
                                response=httpx.Response(429))
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_classify_500_transient():
    """服务端 500 → transient（Ollama 可能恢复）"""
    exc = httpx.HTTPStatusError("500", request=httpx.Request("POST", "http://x"),
                                response=httpx.Response(500))
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_classify_404_permanent():
    """404 → permanent（路径/资源不存在，重试无意义）"""
    exc = httpx.HTTPStatusError("404", request=httpx.Request("POST", "http://x"),
                                response=httpx.Response(404))
    assert classify_error(exc) == ErrorCategory.PERMANENT


def test_classify_value_error_permanent():
    """普通 ValueError → permanent（保守：不确定就不重试）"""
    assert classify_error(ValueError("bad param")) == ErrorCategory.PERMANENT


# ──────────────────────────────────────────────
# 重试策略
# ──────────────────────────────────────────────

def test_retry_limits_attempts():
    """transient 错误在 max_attempts 内重试，超限不再重试"""
    policy = ExponentialBackoff(max_attempts=3)
    assert policy.should_retry(0, ErrorCategory.TRANSIENT) is True
    assert policy.should_retry(1, ErrorCategory.TRANSIENT) is True
    assert policy.should_retry(2, ErrorCategory.TRANSIENT) is True
    assert policy.should_retry(3, ErrorCategory.TRANSIENT) is False  # 超限


def test_retry_rejects_permanent():
    """permanent 错误一律不重试"""
    policy = ExponentialBackoff(max_attempts=3)
    assert policy.should_retry(0, ErrorCategory.PERMANENT) is False


def test_retry_delay_exponential():
    """退避延迟随尝试次数指数增长"""
    policy = ExponentialBackoff(base_delay=2.0, max_delay=60.0)
    d0 = policy.next_delay(0)  # ~2s
    d1 = policy.next_delay(1)  # ~4s
    d2 = policy.next_delay(2)  # ~8s
    assert d0 < d1 < d2
    assert 1.5 <= d0 <= 2.5  # 2.0 * jitter(0.8~1.2)
    assert d2 <= 60.0  # 上限


def test_retry_delay_capped():
    """退避延迟不超过 max_delay"""
    policy = ExponentialBackoff(base_delay=2.0, max_delay=10.0)
    d5 = policy.next_delay(5)  # 2*32=64 → 封顶 10
    assert d5 <= 10.0


# ──────────────────────────────────────────────
# llm_client 重试行为
# ──────────────────────────────────────────────

class FakeResponse:
    """模拟 httpx.Response（同步方法，供 _send_request 使用）"""

    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self) -> dict:
        return self._data


@pytest.mark.asyncio
async def test_llm_retries_transient_then_succeeds():
    """
    transient 错误（连接失败）→ 自动重试 → 第二次成功。
    验证网络闪断不导致整个调用失败。
    """
    from app.infrastructure.llm_client import OllamaChatClient, Message

    client = OllamaChatClient("http://test:11434", "test-model",
                              retry_policy=ExponentialBackoff(max_attempts=3, base_delay=0.01))

    calls = {"count": 0}
    async def fake_post(url, json):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("第一次连接失败")
        return FakeResponse({
            "message": {"role": "assistant", "content": "重试后的回答"},
            "done": True,
        })

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await client.chat([Message(role="user", content="你好")])

    assert result == "重试后的回答"
    assert calls["count"] == 2  # 重试了一次


@pytest.mark.asyncio
async def test_llm_raises_on_permanent_error():
    """
    permanent 错误（如 404）→ 不重试，直接抛出。
    """
    from app.infrastructure.llm_client import OllamaChatClient, Message

    client = OllamaChatClient("http://test:11434", "test-model",
                              retry_policy=ExponentialBackoff(max_attempts=3, base_delay=0.01))

    def fake_post(url, json):
        raise httpx.HTTPStatusError(
            "404", request=httpx.Request("POST", url),
            response=httpx.Response(404),
        )

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat([Message(role="user", content="你好")])


@pytest.mark.asyncio
async def test_llm_gives_up_after_max_attempts():
    """
    transient 错误持续发生 → 重试到上限后仍抛出（不无限重试）。
    """
    from app.infrastructure.llm_client import OllamaChatClient, Message

    client = OllamaChatClient("http://test:11434", "test-model",
                              retry_policy=ExponentialBackoff(max_attempts=3, base_delay=0.01))

    calls = {"count": 0}
    async def fake_post(url, json):
        calls["count"] += 1
        raise httpx.ConnectError("持续连接失败")

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        with pytest.raises(httpx.ConnectError):
            await client.chat([Message(role="user", content="你好")])

    assert calls["count"] == 4  # 1 次原始 + 3 次重试
