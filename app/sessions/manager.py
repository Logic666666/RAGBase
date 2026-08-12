"""
会话管理器

管理会话生命周期：
  submit → 创建记录 + 后台执行 → 完成/失败更新状态

后台执行方式：进程内 asyncio 任务。
  - 优点：实现简单，无需额外组件
  - 代价：会话生命周期 = 进程生命周期（重启丢失执行中会话）
  - 研究报告任务为分钟级，可接受；transcript 已落盘，将来可支持 resume

依赖注入设计：
  SessionManager 并不 import build_agent（避免与 main.py 循环依赖），
  而是通过 agent_builder 回调注入——上层（main.py）决定怎么构建 Agent。
"""

import asyncio
import logging
import uuid
from typing import Callable

from ..agent.orchestrator import Agent
from ..core.config import Settings
from .base import SessionRecord, SessionStatus, SessionStore


class SessionManager:
    """
    异步会话管理器（对齐 Claude Code 的 session 生命周期管理）。

    用法（由 main.py 组装）：
      manager = SessionManager(store=JsonSessionStore(...), agent_builder=build_agent)
      session_id = await manager.submit(kb="x", task="调研...", max_steps=5, settings=settings)
    """

    def __init__(
        self,
        store: SessionStore,
        agent_builder: Callable[[Settings, str], Agent],
        max_sessions: int = 50,
    ):
        """
        Args:
            store:         会话存储（接口注入，实现可替换）
            agent_builder: Agent 构建回调：agent_builder(settings, kb) -> Agent
            max_sessions:  保留的会话数量上限（超出自动清理最旧的）
        """
        self.store = store
        self.agent_builder = agent_builder
        self.max_sessions = max_sessions
        self._sessions: dict[str, asyncio.Task] = {}

    async def submit(
        self,
        kb: str,
        task: str,
        max_steps: int,
        settings: Settings,
    ) -> str:
        """
        提交会话：创建记录并立即返回 session_id，后台执行。

        Returns:
            session_id（调用方用它轮询 status / 获取 result）
        """
        session_id = uuid.uuid4().hex[:8]
        record = SessionRecord(
            session_id=session_id,
            kb=kb,
            task=task,
            max_steps=max_steps,
        )
        await self.store.create(record)

        # 后台执行（不阻塞当前请求）
        self._sessions[session_id] = asyncio.create_task(
            self._run(record, settings)
        )

        # 保留策略：提交后清理超出上限的最旧会话（带日志，非静默）
        try:
            removed = await self.store.cleanup(self.max_sessions)
            if removed:
                logging.info(f"会话保留清理：删除 {removed} 个最旧会话（上限 {self.max_sessions}）")
        except Exception as e:
            logging.warning(f"会话保留清理失败: {e}")

        return session_id

    async def get_status(self, session_id: str) -> SessionRecord | None:
        """查询会话状态（running/done/failed）"""
        return await self.store.get(session_id)

    async def list_sessions(self) -> list[SessionRecord]:
        """列出所有历史会话（按创建时间倒序）——追溯入口"""
        return await self.store.list()

    async def delete_session(self, session_id: str) -> None:
        """删除会话记录及其工作区数据"""
        await self.store.delete(session_id)

    async def _run(self, record: SessionRecord, settings: Settings) -> None:
        """
        后台执行 Agent，完成后更新会话状态。

        状态流转：
          running → done（写入 result/steps/trace/messages）
          running → failed（写入 error + 已产生的 trace）

        transcript 落盘：Agent 循环的完整消息历史
        随会话保存（对齐 Claude Code 的 session transcript），
        为中断恢复（resume）与全程追溯提供数据基础。
        失败会话也保存 trace——能看到失败前执行到哪一步（可复盘）。
        """
        agent = None
        try:
            # 传 session_id + max_steps：build_agent 用它创建工作区并设置步数上限
            agent = self.agent_builder(
                settings, record.kb, record.session_id, record.max_steps
            )
            result = await agent.run(record.task)

            await self.store.update(record.session_id, {
                "status": SessionStatus.DONE,
                "result": result.answer,
                "completed": result.completed,
                "reason": result.reason,
                "steps": result.steps,
                # transcript：完整消息历史（resume/追溯的数据基础）
                "messages": agent.history,
                # trace：执行轨迹（可复盘）
                "trace": agent.tracer.summary() if agent.tracer else None,
            })
        except Exception as e:
            patch = {
                "status": SessionStatus.FAILED,
                "error": f"{type(e).__name__}: {str(e)}",
            }
            # 失败会话也保留 trace：可以看到失败前执行到哪一步
            if agent is not None and agent.tracer:
                patch["trace"] = agent.tracer.summary()
            await self.store.update(record.session_id, patch)
            await self.store.update(record.session_id, {
                "status": SessionStatus.FAILED,
                "error": f"{type(e).__name__}: {str(e)}",
            })
        finally:
            self._sessions.pop(record.session_id, None)
