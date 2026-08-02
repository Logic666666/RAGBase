"""
Trace 记录器

记录 Agent 执行过程中的每一步事件，用于：
  1. 调试——Agent 答错时，回看每步发生了什么
  2. 展示——前端展示 Agent 的思考过程（可复盘）
  3. 评估——离线评测的数据基础

事件类型（由 Agent orchestrator 产生）：
  start        → 任务开始
  llm          → LLM 每轮输出
  tool_call    → Agent 决定调用工具
  tool_result  → 工具返回结果
  final_answer → Agent 给出最终回答
  max_steps    → 步数耗尽被截断
  give_up      → 连续出错主动放弃
"""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class TraceEvent:
    """一次事件的记录"""
    event: str          # 事件类型（见模块 docstring）
    detail: str         # 事件详情（截断后的文本）
    timestamp: float    # 发生时间（epoch 秒）
    step: int = 0       # 第几步（0 表示非循环步骤）


class Tracer:
    """
    执行追踪器。

    用法：
      tracer = Tracer()
      tracer.record("tool_call", "search_kb(query=数据库)")
      summary = tracer.summary()   # → 可 JSON 序列化的 dict
    """

    def __init__(self):
        self.run_id: str = uuid.uuid4().hex[:8]
        self.events: list[TraceEvent] = []
        self._step = 0

    def record(self, event: str, detail: str) -> None:
        """
        记录一个事件。

        自动管理 step 计数：tool_call 视为一步的开始，
        每记录一次 tool_call 就递增步数。
        """
        if event == "tool_call":
            self._step += 1

        self.events.append(
            TraceEvent(
                event=event,
                detail=detail[:500],   # 截断，防止超大事件
                timestamp=time.time(),
                step=self._step,
            )
        )

    def summary(self) -> dict:
        """
        生成执行摘要，可直接 JSON 序列化。

        结构：
        {
            "run_id": "a1b2c3d4",
            "event_count": 6,
            "steps": 2,
            "events": [
                {"event": "start", "detail": "...", "timestamp": ..., "step": 0},
                ...
            ]
        }
        """
        return {
            "run_id": self.run_id,
            "event_count": len(self.events),
            "steps": self._step,
            "events": [
                {
                    "event": e.event,
                    "detail": e.detail,
                    "timestamp": e.timestamp,
                    "step": e.step,
                }
                for e in self.events
            ],
        }
