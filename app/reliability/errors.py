"""
错误分类（对齐 Claude Code 的 error classification）

Claude Code 内部把错误分类为：429 限流、529 容量、连接错误、认证错误等，
并据此决定恢复策略（可恢复 → 退避重试；不可恢复 → 直接报错）。

目前项目策略暂且简化为两类：
  TRANSIENT  临时错误——网络超时/连接失败/服务端 5xx
             重试能解决（Ollama 重启、网络恢复、负载下降）
  PERMANENT  永久错误——参数/逻辑错误
             重试只会重复失败（烧 token），直接报给上层
"""

from enum import Enum

import httpx


class ErrorCategory(str, Enum):
    """错误类别"""
    TRANSIENT = "transient"   # 临时错误：可重试
    PERMANENT = "permanent"   # 永久错误：不可重试


# httpx 网络类异常（连接失败/超时/协议错误）→ 临时错误
_TRANSIENT_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
    httpx.TransportError,
)

# 服务端可恢复的 HTTP 状态码 → 临时错误
_TRANSIENT_STATUS_CODES = (429, 500, 502, 503, 529)


def classify_error(exc: Exception) -> ErrorCategory:
    """
    按异常类型/状态码分类错误。

    Args:
        exc: 捕获的异常（通常来自 httpx 调用）

    Returns:
        ErrorCategory.TRANSIENT（可重试）或 PERMANENT（不可重试）
    """
    # 网络类异常 → 临时
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return ErrorCategory.TRANSIENT

    # HTTP 状态错误（非 2xx）→ 看状态码
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code in _TRANSIENT_STATUS_CODES:
            return ErrorCategory.TRANSIENT
        return ErrorCategory.PERMANENT

    # 其他异常默认永久（保守：不确定就不重试）
    return ErrorCategory.PERMANENT
