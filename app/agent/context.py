"""
上下文管理（对齐 Claude Code 的 context compaction 设计）

解决情境：研究报告任务多轮检索后，工具结果大量累积在消息历史，
导致模型"失忆"（注意力分散）和上下文撑爆。

Claude Code 的做法（context-window 文档）：
  - autoCompact（摘要）+ snipCompact（裁剪）+ contextCollapse（重构）
  - ~70% 上下文占用时触发
  - 已知失败模式：阈值太激进会 thrashing 烧 token；全量摘要丢近期状态

当前取舍：
  - 本次只做"工具结果裁剪"（TrimCompressor）——无信息语义损失：
    来源路径保留（Agent 需要细节时可重新检索），正文截断
  - 消息历史的 LLM 摘要（compact）留接口，后续实现——
    避免全量摘要丢失近期状态的 Claude Code 已知问题

设计原则（接口抽象 + 单一实现）：
  ContextCompressor 是抽象接口，TrimCompressor 是当前实现。
"""

from abc import ABC, abstractmethod


class ContextCompressor(ABC):
    """
    上下文压缩器抽象接口。

    上层（orchestrator）只依赖本接口：
      compress_tool_result：工具结果进入消息历史前的裁剪
      should_compact / compact：消息历史压缩（预留，后续 LLM 摘要实现）
    """

    @abstractmethod
    def compress_tool_result(self, result: str) -> str:
        """
        压缩工具结果文本。

        工具结果（如检索文档片段）在进入消息历史前调用，
        控制单轮结果占用的上下文量。
        """
        ...

    @abstractmethod
    def should_compact(self, messages: list) -> bool:
        """消息历史是否达到压缩阈值（预留扩展点）"""
        ...

    @abstractmethod
    def compact(self, messages: list) -> list:
        """压缩消息历史（预留：LLM 摘要方案后续实现）"""
        ...


class TrimCompressor(ContextCompressor):
    """
    工具结果裁剪（当前方案）。

    策略：超过 max_chars 时截断，但保留开头部分
    ——检索结果的格式是 "[1] (来源路径)\n内容..."，
    来源路径在最前面，截断后依然可追溯、可重查。

    选择截断而不是摘要：
      摘要需要 LLM 调用（慢 + 烧 token + 可能丢关键细节），
      而研究报告场景 Agent 需要细节时可以重新检索——
      "我需要这个工具输出的全部内容，还是只需要结论？"
    """

    def __init__(self, max_chars: int = 2000):
        """
        Args:
            max_chars: 单次工具结果进入上下文的最大字符数
        """
        self.max_chars = max_chars

    def compress_tool_result(self, result: str) -> str:
        if len(result) <= self.max_chars:
            return result
        # 保留开头（来源标记在前），末尾标记截断
        return result[:self.max_chars] + "\n...(结果过长已截断，可重新检索获取细节)"

    def should_compact(self, messages: list) -> bool:
        # 预留：本次不触发消息历史压缩（避免过度设计）
        return False

    def compact(self, messages: list) -> list:
        return messages
