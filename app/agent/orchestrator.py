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

import json

from ..infrastructure.llm_client import Message, OllamaChatClient
from ..tools.base import ToolResult
from ..tools.registry import ToolRegistry
from .context import ContextCompressor, TrimCompressor
from .prompts import PLANNING_PROMPT, system_prompt
from .schemas import AgentResult, parse_tool_call


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
        """
        self.llm = llm
        self.tools = tools
        self.tracer = tracer
        self.max_steps = max_steps
        self.max_parse_failures = max_parse_failures
        self.compressor = compressor or TrimCompressor()
        # 本次执行的消息历史（transcript）：
        # 运行结束后可读取（agent.history），
        # 由 SessionManager 落盘，支持恢复执行（resume）与追溯
        self.history: list[dict] = []

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

        # 1. 规划轮：生成研究计划作为上下文锚点
        # （对齐 Claude Code 的 plan mode：研究先行、执行在后）
        # 计划不强制解析为结构化数据——直接作为文本锚点，
        # 模型在后续循环中始终能看到"我要研究哪些维度"。
        plan = await self._plan(task)
        self._trace("plan", plan)

        # 2. 构建消息历史（研究计划作为 system 消息锚点）
        messages: list[Message] = [
            Message(role="system", content=system_prompt(self.tools.schemas())),
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
    # 内部：规划轮
    # ──────────────────────────────────────────

    async def _plan(self, task: str) -> str:
        """
        规划轮：生成研究计划。

        对齐 Claude Code 的 "Explore → plan → code"：
        先探索项目结构（list_files），再基于实际内容规划维度——
        避免"空规划"（模型不了解代码库，按用户问题的关键词猜维度，
        而实际代码库可能完全没有这些内容）。

        计划不强制 JSON 解析（小模型输出不稳定）——
        直接作为文本锚点使用，模型自己理解。
        """
        # 先探索：获取项目文件结构（若工具不可用则降级为无结构）
        structure = "（无法获取项目结构）"
        if self.tools.get("list_files") is not None:
            structure = await self.tools.execute("list_files", {"pattern": "*"})

        response = await self.llm.chat([
            Message(role="system", content=PLANNING_PROMPT),
            Message(
                role="user",
                content=f"项目文件结构：\n{structure}\n\n研究主题：{task}",
            ),
        ])
        return response.strip()

    # ──────────────────────────────────────────
    # 内部：Trace 记录
    # ──────────────────────────────────────────

    def _trace(self, event: str, detail: str) -> None:
        """记录一步执行（如果提供了 tracer）"""
        if self.tracer:
            self.tracer.record(event, detail)
