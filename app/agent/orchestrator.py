"""
Agent 循环（ReAct Orchestrator）

核心控制循环：
  思考 → 行动（工具调用）→ 观察（工具结果）→ 再思考 → ... → 最终回答

循环逻辑：
  1. 把任务和系统提示词发给 LLM
  2. LLM 返回工具调用 JSON → 执行工具 → 把结果写回消息历史 → 回到 1
  3. LLM 返回回答文本（无 tool 字段）→ 循环终止，这就是最终回答

安全机制：
  - max_steps 限制（防死循环）
  - 连续解析失败限制（防模型反复输出坏 JSON）
  - 工具执行错误以文本返回给 LLM（Agent 自己决定如何应对）

设计原则：
  - orchestrator 不做业务决策，只做循环控制
  - 工具是插拔的（通过 ToolRegistry 注入）
  - 每一步都可通过 Tracer 记录（可观测）
"""

import asyncio
import json
import re
from typing import TYPE_CHECKING

from ..infrastructure.llm_client import Message, OllamaChatClient
from ..tools.base import ToolResult
from ..tools.registry import ToolRegistry
from .context import ContextCompressor, TrimCompressor
from .prompts import PLANNING_PROMPT, system_prompt
from .schemas import AgentResult, parse_tool_call

if TYPE_CHECKING:
    from ..events.bus import EventBus


class Agent:
    """
    ReAct Agent。

    Agent 本身不知道任何业务逻辑——它只做循环控制：
      调用 LLM → 解析结果 → 需要工具就执行 → 继续循环 → 得到回答就返回
    """

    def __init__(
        self,
        llm: OllamaChatClient,
        tools: ToolRegistry,
        tracer=None,
        max_steps: int = 5,
        max_parse_failures: int = 3,
        compressor: ContextCompressor | None = None,
        event_bus=None,
        session_id: str = "",
    ):
        """
        Args:
            llm:             手写的 LLM 客户端
            tools:           工具注册中心
            tracer:          可观测记录器
            max_steps:       最大循环步数（防死循环，默认 5）
            max_parse_failures: 连续解析失败的容忍次数（默认 2）
            compressor:      上下文压缩器（工具结果裁剪）。
                             默认 TrimCompressor，控制单轮结果占用
            event_bus:       事件总线（实时推送步骤事件给前端，可选）
            session_id:      会话标识（事件按会话隔离发布）
        """
        self.llm = llm
        self.tools = tools
        self.tracer = tracer
        self.max_steps = max_steps
        self.max_parse_failures = max_parse_failures
        self.compressor = compressor or TrimCompressor()
        self.event_bus = event_bus
        self.session_id = session_id
        # 本次执行的消息历史（transcript）：
        # 运行结束后可读取（agent.history），
        # 由 SessionManager 落盘，支持恢复执行（resume）与追溯
        self.history: list[dict] = []
        # 提前收尾质疑标志（仅质疑一次，防死循环）
        self._final_questioned = False
        # 收尾确认标志（仅要求一次纯文本回答，防死循环）
        self._final_asked = False
        # 文件清单总数（_plan 阶段从 list_files 结果解析，供收尾质疑）
        self._total_files = 0

    async def run(self, task: str) -> AgentResult:
        """
        执行一个任务。

        流程：
        1. 构建消息历史（system prompt + 用户任务）
        2. 进入循环（最多 max_steps 步）：
           a. LLM 推理
           b. 解析输出：工具调用 or 最终回答
           c. 工具调用 → 执行 → 结果写回消息历史 → 继续循环
           d. 最终回答 → 返回结果

        Args:
            task: 用户的任务描述

        Returns:
            AgentResult(answer, completed, steps)
        """
        self._trace("start", task)

        # 1. 规划轮：先探索项目结构，再生成研究计划作为上下文锚点
        # （对齐 Claude Code 的 plan mode：研究先行、执行在后）
        # 计划不强制解析为结构化数据，直接将文本作为锚点，
        # 模型在后续循环中始终能看到需要研究的维度。
        structure, plan = await self._plan(task)
        self._trace("plan", plan)

        # 文件清单总数（供提前收尾质疑）：
        # list_files 输出格式稳定（"共 N 个文件：..."），解析失败则质疑不触发
        m = re.search(r"共 (\d+) 个文件", structure)
        self._total_files = int(m.group(1)) if m else 0

        # 2. 构建消息历史：
        #   探索结果 + 研究计划作为 system 消息锚点—
        #   主循环模型能直接看到"项目结构已探索"，不会重复调 list_files
        messages: list[Message] = [
            Message(role="system", content=system_prompt(self.tools.schemas())),
            Message(role="system", content=f"项目文件结构（已探索，无需重复查看）：\n{structure}"),
            Message(role="system", content=f"研究计划：\n{plan}"),
            Message(role="user", content=task),
        ]

        parse_failures = 0
        # 收敛检测（对齐 Claude Code 的 convergence detection——Issue #30150：
        #   模型重复相同工具调用/输出导致死循环无进展）
        # 1) 工具级：记录已用过的 (tool, arguments)，拦截重复工具调用
        seen_calls: set[tuple[str, str]] = set()
        # 2) 响应级：记录上一轮 LLM 输出，拦截"复读自身"（echo mode——
        #    小模型在循环中复制自己上一轮的整段输出，工具级检测拦不住）
        last_response: str | None = None

        # 2. 循环（try/finally 保证任何退出路径都保存 transcript）
        try:
            for step in range(self.max_steps):
                # 上下文管理：消息历史超阈值时折叠早期轮次为摘要
                # （对齐 Claude Code 的 autoCompact——每次 API 调用前检查）
                if self.compressor.should_compact(messages):
                    new_messages = await self.compressor.compact(messages)
                    # 仅在实际折叠时记录事件：
                    # compact 可能因轮次不足/摘要失败返回原消息（降级契约），
                    # 此时记录"已折叠"会误导观测
                    if new_messages is not messages:
                        self._trace("compact", f"折叠早期轮次（保留最近 {self.compressor.keep_recent_rounds if hasattr(self.compressor, 'keep_recent_rounds') else ''} 轮）")
                    messages = new_messages

                # 2a. LLM 推理
                chat_response = await self.llm.chat_with_response(messages)
                response = chat_response.content

                # 记录思考过程（模型推理内容）：
                # Ollama 通过 message.thinking 或 <think> 标签返回，
                # llm_client 已提取到 ChatResponse.thinking。
                # 推理过程是评估 agent 质量的重要数据，必须保留在 trace 中。
                if chat_response.thinking:
                    self._trace("think", chat_response.thinking)

                # 记录 LLM 正文输出（不含 think 标签）：
                # 前端需要从 llm 事件解析 thought 字段和工具调用 JSON
                self._trace("llm", response)

                # 记录结束原因（诊断输出截断的关键）：
                # done_reason = "length" 说明达到 num_predict 上限（输出被截断）
                if chat_response.done_reason != "stop":
                    self._trace("llm_done", f"done_reason={chat_response.done_reason}")

                # 响应级收敛检测：输出与上一轮逐字相同 → 模型在复读自己（无进展）
                if last_response is not None and response == last_response:
                    result = ToolResult.error(
                        "你刚才的输出与上一轮完全相同，说明没有进展。"
                        "请改变策略，或基于已有信息直接输出最终回答。"
                    )
                    self._trace("tool_result", result.content)
                    parse_failures += 1
                    if parse_failures >= self.max_parse_failures:
                        self._trace("give_up", "连续无进展，终止循环")
                        return AgentResult(
                            answer="Agent 连续输出相同内容（无进展），任务未能完成。"
                                   f"最后一次错误: {result.content[:200]}",
                            completed=False,
                            steps=step + 1,
                            reason="tool_errors",
                        )
                    # 写回消息历史后继续循环
                    messages.append(Message(role="assistant", content=response))
                    messages.append(
                        Message(role="user", content=f"[工具错误]\n{result.content}")
                    )
                    continue
                last_response = response

                # 2b. 解析输出
                tool_call = parse_tool_call(response)

                if tool_call is None:
                    # 判定为最终回答
                    self._trace("final_answer", response)
                    return AgentResult(answer=response, completed=True, steps=step + 1)

                # 空工具名 = 模型输出无效 JSON：
                # 区分两种语义——
                # 1) tool=null/空 但 thought 有内容：模型在"思考后准备结束"
                #    （thought 如"已有足够信息，无需再调工具"）
                #    → 视为最终回答，返回 thought 内容（语义容错）
                # 2) 完全退化（空 thought + 空工具）→ 报错并计入失败计数
                if tool_call.tool in (None, "", "None", "null"):
                    if tool_call.thought:
                        # 提前收尾质疑（分级）：
                        # 小模型常见退化——未读完清单中的文件就想收尾。
                        # 0 阅读 = 硬错误：未读任何文件就"分析所有文章"必然编造，
                        #   不给"信息已足够"退路，计入失败计数防死循环；
                        # 部分阅读 = 软质疑（仅一次）：读了部分可以诚实收尾
                        #   （报告须说明未读原因），第二次收尾即接受
                        read_count = self._read_file_count(messages)
                        if (
                            self._total_files > 0
                            and read_count < self._total_files
                        ):
                            if read_count == 0:
                                hint = (
                                    "你尚未阅读任何文件。"
                                    "任务要求分析全部文档，未阅读即输出报告"
                                    "会基于编造内容。请先阅读文件，"
                                    "再基于实际内容输出报告。"
                                )
                                self._trace("final_question", hint)
                                messages.append(Message(role="assistant", content=response))
                                messages.append(Message(role="user", content=f"[任务提示]\n{hint}"))
                                parse_failures += 1
                                if parse_failures >= self.max_parse_failures:
                                    self._trace("give_up", "连续未阅读即收尾，终止循环")
                                    return AgentResult(
                                        answer="Agent 未阅读任何文件就试图结束，任务未能完成。"
                                               "最后一次错误: " + hint,
                                        completed=False,
                                        steps=step + 1,
                                        reason="tool_errors",
                                    )
                                continue
                            if not self._final_questioned:
                                self._final_questioned = True
                                hint = (
                                    f"你已阅读 {read_count}/{self._total_files} 个文件，"
                                    "任务尚未覆盖清单中的全部文档。"
                                    "若信息已足够，请直接输出完整最终报告"
                                    "（报告中需说明未阅读文件及原因）；"
                                    "否则请继续阅读剩余文件。"
                                )
                                self._trace("final_question", hint)
                                messages.append(Message(role="assistant", content=response))
                                messages.append(Message(role="user", content=f"[任务提示]\n{hint}"))
                                continue
                        # 收尾确认（仅一次）：
                        # 模型常把"收尾声明"（thought=我打算输出报告）当输出，
                        # 直接接受会把声明当回答，任务"完成"了却没有报告。
                        # 要求纯文本回答——意图正确时模型下一轮即输出正文；
                        # 第二次收尾才接受（有界，防死循环）
                        if not self._final_asked:
                            self._final_asked = True
                            hint = (
                                "你已结束工具调用。请直接以纯文本输出"
                                "最终回答/报告正文（不要输出 JSON 或工具调用格式）。"
                            )
                            self._trace("final_question", hint)
                            messages.append(Message(role="assistant", content=response))
                            messages.append(Message(role="user", content=f"[任务提示]\n{hint}"))
                            continue
                        # 模型表达"不需要更多工具"——收尾思考即回答
                        self._trace("final_answer", tool_call.thought)
                        return AgentResult(
                            answer=tool_call.thought,
                            completed=True,
                            steps=step + 1,
                        )
                    self._trace("tool_call", f"(空工具名) {response[:100]}")
                    result = ToolResult.error(
                        "工具名为空（JSON 格式异常）。"
                        "请重新生成正确的工具调用，或基于已有信息直接输出最终回答。"
                    )
                    self._trace("tool_result", result.content)
                    parse_failures += 1
                    if parse_failures >= self.max_parse_failures:
                        self._trace("give_up", "连续工具错误，终止循环")
                        return AgentResult(
                            answer="工具调用连续出错，任务未能完成。"
                                   f"最后一次错误: {result.content[:200]}",
                            completed=False,
                            steps=step + 1,
                            reason="tool_errors",
                        )
                    messages.append(Message(role="assistant", content=response))
                    messages.append(
                        Message(role="user", content=f"[工具错误]\n{result.content}")
                    )
                    continue

                # 2c. 执行工具
                # 记录工具名和参数（前端展示为工具调用行）
                self._trace("tool_call", f"{tool_call.tool} {tool_call.arguments}")

                # 收敛检测：相同 (工具, 参数) 重复调用 → 提示换关键词/换工具
                call_key = (
                    tool_call.tool,
                    json.dumps(tool_call.arguments, sort_keys=True, ensure_ascii=False),
                )
                if call_key in seen_calls:
                    # 只报事实，不引导具体工具（工具选择依据见系统提示中的工具选择指南）
                    if tool_call.tool == "list_files":
                        # list_files 特化：结构已探索且结果在消息历史中，
                        # 重复列出不会产生新信息——引导转向阅读（防空转）
                        result = ToolResult.error(
                            "文件列表已存在于消息历史（文件结构锚点），"
                            "重复列出不会产生新信息。"
                            "请基于清单直接阅读文件，再输出报告。"
                        )
                    else:
                        result = ToolResult.error(
                            f"重复调用：你已用相同参数调用过 {tool_call.tool}，"
                            f"反复相同调用不会产生新结果。"
                            f"请参照工具选择指南更换策略后重试。"
                        )
                else:
                    seen_calls.add(call_key)
                    result = await self.tools.execute(tool_call.tool, tool_call.arguments)
                self._trace("tool_result", result.content)

                # 上下文管理：工具结果进入消息历史前裁剪
                # （对齐 Claude Code 的 tool output 压缩——
                #   来源保留可重查，正文截断控制上下文占用）
                tool_content = self.compressor.compress_tool_result(result.content)

                # 把 LLM 的工具调用和工具结果写回消息历史
                # 成功/失败用不同前缀（对应 Claude Code 的 tool_result + is_error 标记），
                # 模型能明确区分结果与错误
                messages.append(Message(role="assistant", content=response))
                if result.ok:
                    messages.append(
                        Message(
                            role="user",
                            content=f"[工具结果 {tool_call.tool}]\n{tool_content}",
                        )
                    )
                else:
                    messages.append(
                        Message(
                            role="user",
                            content=f"[工具错误 {tool_call.tool}]\n{tool_content}",
                        )
                    )

                # 工具调用失败（参数校验错误/工具不存在）→ 计数防止死循环
                if not result.ok:
                    parse_failures += 1
                    if parse_failures >= self.max_parse_failures:
                        self._trace("give_up", "连续工具错误，终止循环")
                        return AgentResult(
                            answer="工具调用连续出错，任务未能完成。"
                                   f"最后一次错误: {result.content[:200]}",
                            completed=False,
                            steps=step + 1,
                            reason="tool_errors",
                        )

            # 3. 达到 max_steps
            self._trace("max_steps", f"达到最大步数 {self.max_steps}")
            return AgentResult(
                answer=f"已达最大步数（{self.max_steps}），任务可能未完成。"
                       f"请尝试将任务分解为更小的步骤。",
                completed=False,
                steps=self.max_steps,
                reason="max_steps",
            )
        finally:
            # transcript：把消息历史转为可序列化 dict 列表
            # 对齐 Claude Code 的 session transcript——任何退出路径都保留
            self.history = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]

    # ──────────────────────────────────────────
    # 内部：提前收尾质疑辅助
    # ──────────────────────────────────────────

    def _read_file_count(self, messages: list) -> int:
        """
        已成功阅读的文件数：
          - 当前消息中的 read_pdf / read_file 成功结果
          - 摘要消息中的"已阅读文件"清单（折叠轮次的文件已被折叠为摘要，
            清单是它们的唯一记录——否则压缩后计数骤降，质疑会误报）

        仅报事实（读过几个文件），供提前收尾质疑判断；
        不解析任务语义——"没读完"是事实，"必须读完"由模型/提示词决定。
        """
        count = sum(
            1 for m in messages
            if m.role == "user"
            and (
                m.content.startswith("[工具结果 read_pdf]")
                or m.content.startswith("[工具结果 read_file]")
            )
        )
        for m in messages:
            if m.role == "system" and m.content.startswith("研究进展摘要"):
                for line in m.content.splitlines():
                    if line.startswith("已阅读文件："):
                        count += len([p for p in line[len("已阅读文件："):].split("、") if p])
        return count

    # ──────────────────────────────────────────
    # 内部：规划轮
    # ──────────────────────────────────────────

    async def _plan(self, task: str) -> tuple[str, str]:
        """
        规划轮：先探索项目结构，再生成研究计划。

        对齐 Claude Code 的 "Explore → plan → code"：
        先探索项目结构（list_files），再基于实际内容规划维度——
        避免"空规划"（模型不了解代码库，按用户问题的关键词猜维度，
        而实际代码库可能完全没有这些内容）。

        探索结果返回给调用方，作为主循环消息锚点
        （避免主循环模型不知道"结构已探索"，重复调 list_files）。

        计划不强制 JSON 解析（小模型输出不稳定）——
        直接作为文本锚点使用，模型自己理解。

        Returns:
            (structure, plan)：项目文件结构 + 研究计划文本
        """
        # 先探索：获取项目文件结构（若工具不可用则降级为无结构）
        # 与主循环一致：工具调用用统一的 tool_call/tool_result 事件记录
        structure = "（无法获取项目结构）"
        if self.tools.get("list_files") is not None:
            self._trace("tool_call", "list_files {'pattern': '*'}")
            result = await self.tools.execute("list_files", {"pattern": "*"})
            structure = result.content if result.ok else "（无法获取项目结构）"
            self._trace("tool_result", structure[:500])

        response = await self.llm.chat([
            Message(role="system", content=PLANNING_PROMPT),
            Message(
                role="user",
                content=f"项目文件结构：\n{structure}\n\n研究主题：{task}",
            ),
        ])
        return structure, response.strip()

    # ──────────────────────────────────────────
    # 内部：Trace 记录 + 事件发布
    # ──────────────────────────────────────────

    def _trace(self, event: str, detail: str) -> None:
        """
        记录一步执行（单一事件源）：
        1. Tracer（权威记录，落盘）
        2. EventBus（实时推送，前端流式展示）
        两者同源，不重复埋点。
        """
        if self.tracer:
            self.tracer.record(event, detail)
        if self.event_bus and self.session_id:
            # publish 是 async（未来 Redis 实现需要网络 I/O），
            # _trace 是同步埋点——用 create_task 调度（fire-and-forget）
            asyncio.create_task(
                self.event_bus.publish(self.session_id, {
                    "event": event,
                    "detail": detail,
                })
            )
