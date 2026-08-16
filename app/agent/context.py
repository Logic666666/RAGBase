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

from ..infrastructure.llm_client import Message


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
        """消息历史是否达到压缩阈值"""
        ...

    @abstractmethod
    async def compact(self, messages: list) -> list:
        """压缩消息历史（折叠早期轮次为摘要）"""
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
        # 兜底实现：不主动压缩（只做工具结果裁剪）
        return False

    async def compact(self, messages: list) -> list:
        return messages


# 摘要生成提示词
_SUMMARIZE_PROMPT = """\
你是研究助手。以下是 Agent 早期检索轮次的记录（工具调用与结果）。
请总结为一段"研究进展摘要"，要求：
- 包含已收集的关键信息、来源路径、当前已确认的结论
- 简洁（200 字内），保留关键事实
- 只输出摘要文本，不要其他内容"""


class SummaryCompressor(ContextCompressor):
    """
    消息历史摘要压缩（对齐 Claude Code 的 autoCompact）。

    解决情境：研究报告多轮检索后，早期轮次（工具调用+结果）大量累积，
    上下文被占满、模型注意力被稀释。

    策略：
      1. 阈值宽裕（max_rounds 轮后才触发）——避免过度压缩 thrashing
      2. 只折叠早期轮次，保留最近 keep_recent_rounds 轮原文——
         避免全量摘要丢失近期状态（Issue #58749）
      3. system 锚点（工具列表/文件结构/研究计划）始终保留

    折叠后的早期轮次由 LLM 生成一段"研究进展摘要"，
    插入到保留的最近轮次之前。
    """

    def __init__(
        self,
        llm,
        max_rounds: int = 8,
        keep_recent_rounds: int = 3,
        max_chars: int = 8000,
    ):
        """
        Args:
            llm:              用于生成摘要的 LLM 客户端
            max_rounds:       超过 N 轮检索触发压缩（宽裕阈值防 thrashing）
            keep_recent_rounds: 保留最近 N 轮原文（防全量摘要丢近期状态）
            max_chars:        工具结果裁剪阈值（委托给内部 TrimCompressor）
        """
        # 组合：单轮裁剪委托给 TrimCompressor
        self._trim = TrimCompressor(max_chars=max_chars)
        self.llm = llm
        self.max_rounds = max_rounds
        self.keep_recent_rounds = keep_recent_rounds

    def compress_tool_result(self, result: str) -> str:
        """单轮工具结果裁剪（委托 TrimCompressor）"""
        return self._trim.compress_tool_result(result)

    def should_compact(self, messages: list) -> bool:
        """按工具结果轮数判断是否触发压缩"""
        tool_results = [
            m for m in messages
            if m.role == "user" and m.content.startswith("[工具结果")
        ]
        return len(tool_results) > self.max_rounds

    async def compact(self, messages: list) -> list:
        """
        折叠早期轮次为摘要，保留最近 N 轮原文 + 全部 system 锚点。

        消息结构：system 锚点 + user(任务) + [assistant+user] 轮次对
        返回：system 锚点 + user(任务) + 摘要 + 最近 N 轮
        """
        system_msgs = [m for m in messages if m.role == "system"]
        others = [m for m in messages if m.role != "system"]

        if not others:
            return messages
        task = others[0]                    # user(任务)，保留
        rounds = others[1:]                 # assistant+user 轮次对

        # 轮次数不足（含最近保留 + 至少一轮可折叠）→ 不压缩
        keep_count = self.keep_recent_rounds * 2  # assistant+user 成对
        if len(rounds) <= keep_count + 2:
            return messages

        fold_msgs = rounds[:-keep_count]   # 早期轮次（折叠）
        recent = rounds[-keep_count:]      # 最近 N 轮（保留原文）

        summary = await self._summarize(fold_msgs)

        return (
            system_msgs
            + [task]
            + [Message(role="system", content=f"研究进展摘要（早期检索已折叠）：\n{summary}")]
            + recent
        )

    async def _summarize(self, messages: list) -> str:
        """用 LLM 把早期轮次总结为摘要"""
        content = "\n".join(
            f"[{m.role}]\n{m.content[:500]}" for m in messages
        )
        response = await self.llm.chat([
            Message(role="system", content=_SUMMARIZE_PROMPT),
            Message(role="user", content=content),
        ])
        return response.strip()