"""
上下文管理（对齐 Claude Code 的 context compaction 设计）

解决情境：研究报告任务多轮检索后，工具结果大量累积在消息历史，
导致模型"失忆"（注意力分散）和上下文撑爆。

Claude Code 的做法（context-window 文档）：
  - autoCompact（摘要）+ snipCompact（裁剪）+ contextCollapse（重构）
  - 按 token 量估算触发（~70% 上下文占用），不按轮数
  - 已知失败模式：阈值太激进会 thrashing 烧 token；全量摘要丢近期状态

当前取舍：
  - TrimCompressor：单轮工具结果裁剪（无信息语义损失——
    来源路径保留，Agent 需要细节时可重新检索）
  - SummaryCompressor：消息历史摘要压缩（折叠早期轮次为摘要，
    保留最近 N 轮原文，摘要继承旧摘要）

设计原则（接口抽象 + 单一实现）：
  ContextCompressor 是抽象接口，TrimCompressor / SummaryCompressor 是两个实现。
"""

import json
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


# 摘要消息统一前缀：compact 时据此识别并继承旧摘要
_SUMMARY_PREFIX = "研究进展摘要（早期检索已折叠）"

# 摘要生成提示词
_SUMMARIZE_PROMPT = """\
你是研究助手。以下是 Agent 早期检索轮次的记录（工具调用与结果），
其中可能包含之前生成的研究进展摘要。
请总结为一段新的"研究进展摘要"，要求：
- 若输入中包含之前的摘要，必须保留其中已确认的关键信息（来源路径、结论），并与新记录合并——不要丢失早期结论
- 包含已收集的关键信息、来源路径、当前已确认的结论
- 简洁（200 字内），保留关键事实
- 只输出摘要文本，不要其他内容"""


class SummaryCompressor(ContextCompressor):
    """
    消息历史摘要压缩（对齐 Claude Code 的 autoCompact）。

    解决情境：研究报告多轮检索后，早期轮次（工具调用+结果）大量累积，
    上下文被占满、模型注意力被稀释。

    策略：
      1. 按字符总量触发（不是轮数）——轮数与上下文占用无直接关系，
         Claude Code / LangChain 均按 token 量估算触发；
         阈值是轮次部分的字符预算（≈ 窗口的 60%，其余留给
         system 锚点与模型输出，按 1 字符 ≈ 1 token 估算中文）
      2. 增量节流：压缩后至少新增 keep_recent_rounds 轮才再次触发，
         防止"压缩后残余仍超阈值 → 立即再压缩"的 thrashing 循环
         （对齐 Claude Code 的 circuit breaker）
      3. 只折叠早期轮次，保留最近 keep_recent_rounds 轮原文——
         避免全量摘要丢失近期状态（Issue #58749）；
         keep 与窗口匹配：残余 ≈ 摘要 + keep × 单轮上限 ≤ 输入预算
      4. 摘要继承：新摘要 = LLM(旧摘要 + 新轮次)，多次压缩不丢早期结论
         （对齐 LangChain 的 predict_new_summary）
      5. 降级：摘要生成失败返回原消息，压缩不阻塞主任务
      6. system 锚点（工具列表/文件结构/研究计划）始终保留
    """

    def __init__(
        self,
        llm,
        trigger_chars: int = 9000,
        keep_recent_rounds: int = 1,
        max_chars: int = 8000,
    ):
        """
        Args:
            llm:              用于生成摘要的 LLM 客户端
            trigger_chars:    轮次部分字符总量触发阈值。默认 9000 ≈
                              16K 窗口输入预算（16384×0.6≈9800），
                              换窗口按 num_ctx×0.6 调整（build_agent 已联动）
            keep_recent_rounds: 保留最近 N 轮原文。默认 1：16K 窗口下
                              残余（摘要 + 1 轮原文 + 锚点 ≈ 10K token）
                              已近上限，keep=3 会直接超窗
            max_chars:        工具结果裁剪阈值（委托给内部 TrimCompressor）
        """
        # 组合：单轮裁剪委托给 TrimCompressor
        self._trim = TrimCompressor(max_chars=max_chars)
        self.llm = llm
        self.trigger_chars = trigger_chars
        self.keep_recent_rounds = keep_recent_rounds
        # 节流基线：上次压缩时的轮次数（新增 ≥ keep 轮才再次触发）
        self._last_compact_rounds = 0

    def compress_tool_result(self, result: str) -> str:
        """单轮工具结果裁剪（委托 TrimCompressor）"""
        return self._trim.compress_tool_result(result)

    def should_compact(self, messages: list) -> bool:
        """轮次部分字符总量超阈值 + 自上次压缩新增足够轮次 → 触发"""
        _, _, _, rounds = self._partition(messages)
        total = sum(len(m.content) for m in rounds)
        return (
            total > self.trigger_chars
            and len(rounds) - self._last_compact_rounds >= self.keep_recent_rounds
        )

    async def compact(self, messages: list) -> list:
        """
        折叠早期轮次为摘要，保留最近 N 轮原文 + 全部 system 锚点。
        旧摘要（若有）并入折叠集合一起总结——摘要继承，不丢早期结论。

        消息结构：system 锚点 + user(任务) + [assistant+user] 轮次对
        返回：system 锚点 + user(任务) + 新摘要 + 最近 N 轮

        降级契约：摘要生成失败返回原消息（本轮不压缩），
        主任务不受影响；节流基线不更新，下次循环可重试。
        """
        system_msgs, old_summary, task, rounds = self._partition(messages)
        if task is None:
            return messages

        # 轮次数不足（含最近保留 + 至少一轮可折叠）→ 不压缩
        keep_count = self.keep_recent_rounds * 2  # assistant+user 成对
        if len(rounds) <= keep_count + 2:
            return messages

        try:
            fold_msgs = rounds[:-keep_count]   # 早期轮次（折叠）
            recent = rounds[-keep_count:]      # 最近 N 轮（保留原文）
            # 摘要继承：旧摘要剥离清单行后传入 LLM（只继承正文）。
            # 已读清单由代码统一维护——若清单依赖 LLM 复述，
            # 模型遗漏复述会导致已读计数丢失（压缩后误触发"未读完"质疑）
            old_text = self._strip_summary_files(old_summary.content) if old_summary else ""
            history = (
                [Message(role="system", content=old_text)] + fold_msgs
                if old_text else fold_msgs
            )
            summary = await self._summarize(history)
            # 全量已读清单 = 旧清单（代码解析旧摘要）+ 新折叠轮（代码提取），
            # 去重保序——压缩后模型据此知道读过什么，
            # 避免重读被折叠的文档与收敛检测冲突（重复调用 → give_up）
            read_files = (
                self._parse_summary_files(old_summary.content) if old_summary else []
            )
            for p in self._read_files(fold_msgs):
                if p not in read_files:
                    read_files.append(p)
            content = f"{_SUMMARY_PREFIX}：\n"
            if read_files:
                content += f"已阅读文件：{'、'.join(read_files)}\n"
            content += summary
            # 节流基线：以压缩前轮次数为准（新增 keep 轮后才再次触发）
            self._last_compact_rounds = len(rounds)
        except Exception:
            # 摘要失败 → 本轮不压缩（下次循环再试），主任务不受影响
            return messages

        return (
            system_msgs
            + [task]
            + [Message(role="system", content=content)]
            + recent
        )

    def _read_files(self, messages: list) -> list:
        """
        提取折叠轮次中成功阅读的文件路径（去重保序）。

        只认成功调用：assistant 调用后必须紧跟 [工具结果（成功）消息，
        [工具错误（如重复调用被拦）不计入——否则被拦的失败调用也会
        进清单，出现"已阅读文件：A、A"的重复。
        """
        paths = []
        for i, m in enumerate(messages):
            if m.role != "assistant":
                continue
            # 配对的下一条必须是成功结果（[工具结果 前缀）
            if i + 1 >= len(messages) or not messages[i + 1].content.startswith("[工具结果"):
                continue
            try:
                data = json.loads(m.content)
            except Exception:
                continue
            if data.get("tool") in ("read_pdf", "read_file"):
                path = (data.get("arguments") or {}).get("path")
                if path and path not in paths:
                    paths.append(path)
        return paths

    def _parse_summary_files(self, content: str) -> list:
        """从摘要文本中提取已读文件清单（代码级继承，不依赖 LLM 复述）"""
        files = []
        for line in content.splitlines():
            if line.startswith("已阅读文件："):
                for p in line[len("已阅读文件："):].split("、"):
                    if p and p not in files:
                        files.append(p)
        return files

    def _strip_summary_files(self, content: str) -> str:
        """剥离摘要文本中的"已阅读文件"行（清单由代码维护，不重复进 LLM 输入）"""
        lines = [l for l in content.splitlines() if not l.startswith("已阅读文件：")]
        return "\n".join(lines).strip()

    def _partition(self, messages: list) -> tuple:
        """
        拆分消息历史：system 锚点 / 旧摘要 / 任务 / 轮次对。

        should_compact 与 compact 共用同一口径，避免两处统计不一致。
        """
        system_msgs = [
            m for m in messages
            if m.role == "system" and not m.content.startswith(_SUMMARY_PREFIX)
        ]
        old_summary = next(
            (m for m in messages
             if m.role == "system" and m.content.startswith(_SUMMARY_PREFIX)),
            None,
        )
        others = [m for m in messages if m.role != "system"]
        task = others[0] if others else None
        rounds = others[1:] if task is not None else []
        return system_msgs, old_summary, task, rounds

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