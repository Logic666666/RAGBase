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

from ..infrastructure.llm_client import Message, OllamaChatClient
from ..tools.registry import ToolRegistry
from .prompts import system_prompt
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
        max_parse_failures: int = 2,
    ):
        """
        Args:
            llm:             手搓的 LLM 客户端
            tools:           工具注册中心
            tracer:          可观测记录器
            max_steps:       最大循环步数（防死循环，默认 5）
            max_parse_failures: 连续解析失败的容忍次数（默认 2）
        """
        self.llm = llm
        self.tools = tools
        self.tracer = tracer
        self.max_steps = max_steps
        self.max_parse_failures = max_parse_failures

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

        # 1. 构建消息历史
        messages: list[Message] = [
            Message(role="system", content=system_prompt(self.tools.schemas())),
            Message(role="user", content=task),
        ]

        parse_failures = 0

        # 2. 循环
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

            # 2b. 解析输出
            tool_call = parse_tool_call(response)

            if tool_call is None:
                # 判定为最终回答
                self._trace("final_answer", response)
                return AgentResult(answer=response, completed=True, steps=step + 1)

            # 2c. 执行工具
            # 记录工具名和参数（前端展示为工具调用行）
            self._trace("tool_call", f"{tool_call.tool} {tool_call.arguments}")
            result = await self.tools.execute(tool_call.tool, tool_call.arguments)
            self._trace("tool_result", result.content)

            # 把 LLM 的工具调用和工具结果写回消息历史
            # 成功/失败用不同前缀（对应 Claude Code 的 tool_result + is_error 标记），
            # 模型能明确区分结果与错误
            messages.append(Message(role="assistant", content=response))
            if result.ok:
                messages.append(
                    Message(
                        role="user",
                        content=f"[工具结果 {tool_call.tool}]\n{result.content}",
                    )
                )
            else:
                messages.append(
                    Message(
                        role="user",
                        content=f"[工具错误 {tool_call.tool}]\n{result.content}",
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
                    )

        # 3. 达到 max_steps
        self._trace("max_steps", f"达到最大步数 {self.max_steps}")
        return AgentResult(
            answer=f"已达最大步数（{self.max_steps}），任务可能未完成。"
                   f"请尝试将任务分解为更小的步骤。",
            completed=False,
            steps=self.max_steps,
        )

    # ──────────────────────────────────────────
    # 内部：Trace 记录
    # ──────────────────────────────────────────

    def _trace(self, event: str, detail: str) -> None:
        """记录一步执行（如果提供了 tracer）"""
        if self.tracer:
            self.tracer.record(event, detail)
