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
    工具结果裁剪（兜底方案，对齐 Claude Code 的 tool output 设计）。

    设计原则（借鉴 Claude Code）：
      1. 阈值要足够大，只做"安全兜底"，不做主动压缩——
         Claude Code 的截断阈值以万字符计（Bash 30K / MCP 25K token），
         过小的阈值会截断模型必须看到的结构化信息（如文件列表）
      2. 截断提示必须揭示总量——模型需要知道"结果有 N 字符，只看到前 M"，
         否则会误以为读到完整内容（Claude Code：completeness misjudgment）
      3. 输出量控制是工具的职责（read_file 的 max_chars、grep 的 max_results），
         orchestrator 的压缩只做最后兜底
    """

    def __init__(self, max_chars: int = 8000):
        """
        Args:
            max_chars: 单次工具结果进入上下文的最大字符数（兜底阈值）
        """
        self.max_chars = max_chars

    def compress_tool_result(self, result: str) -> str:
        if len(result) <= self.max_chars:
            return result
        # 保留开头，截断提示必须包含总量——模型据此判断是否需要重新获取
        total = len(result)
        return (
            result[:self.max_chars]
            + f"\n...(结果已截断：共 {total} 字符，仅显示前 {self.max_chars}。"
            + "如需完整内容，请重新调用工具获取)"
        )

    def should_compact(self, messages: list) -> bool:
        # 预留：本次不触发消息历史压缩（避免过度设计）
        return False

    def compact(self, messages: list) -> list:
        return messages
