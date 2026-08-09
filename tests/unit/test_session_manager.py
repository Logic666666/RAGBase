"""
会话管理器单元测试

验证：提交 → 后台执行 → 状态流转（running → done / failed）→ transcript 落盘。
用 mock agent_builder，不调用真实 Agent。
"""

import asyncio

import pytest

from app.sessions import JsonSessionStore, SessionManager, SessionStatus


class FakeAgent:
    """mock Agent，模拟一个需要时间的会话"""

    def __init__(self, result, delay=0.05, fail=False, history=None):
        self.result = result
        self.delay = delay
        self.fail = fail
        self.tracer = None
        # transcript：模拟 Agent 循环产生的消息历史
        self.history = history or [
            {"role": "system", "content": "你是研究助手"},
            {"role": "user", "content": "调研向量数据库"},
        ]

    async def run(self, task):
        from app.agent.schemas import AgentResult
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("模拟执行失败")
        return AgentResult(answer=self.result, completed=True, steps=2)


def make_agent_builder(agent):
    """构造 mock agent_builder：忽略参数，返回固定 agent"""
    def builder(settings, kb, session_id=None):
        return agent
    return builder


@pytest.mark.asyncio
async def test_submit_and_complete(tmp_path):
    """正常流程：提交 → 轮询到 done → 结果 + transcript 正确"""
    store = JsonSessionStore(str(tmp_path / "sessions"))
    agent = FakeAgent(result="调研完成", delay=0.05)
    manager = SessionManager(store=store, agent_builder=make_agent_builder(agent))

    session_id = await manager.submit(kb="test", task="调研向量数据库", max_steps=5, settings=None)

    # 提交后立即是 running
    record = await manager.get_status(session_id)
    assert record.status == SessionStatus.RUNNING

    # 等待后台完成
    for _ in range(50):
        record = await manager.get_status(session_id)
        if record.status == SessionStatus.DONE:
            break
        await asyncio.sleep(0.02)

    assert record.status == SessionStatus.DONE
    assert record.result == "调研完成"
    assert record.completed is True
    # transcript 落盘验证：消息历史完整保存
    assert len(record.messages) == 2
    assert record.messages[0]["role"] == "system"


@pytest.mark.asyncio
async def test_submit_and_fail(tmp_path):
    """失败流程：Agent 抛异常 → 状态 failed + error 记录"""
    store = JsonSessionStore(str(tmp_path / "sessions"))
    agent = FakeAgent(result="", delay=0.01, fail=True)
    manager = SessionManager(store=store, agent_builder=make_agent_builder(agent))

    session_id = await manager.submit(kb="test", task="会失败的任务", max_steps=5, settings=None)

    for _ in range(50):
        record = await manager.get_status(session_id)
        if record.status == SessionStatus.FAILED:
            break
        await asyncio.sleep(0.02)

    assert record.status == SessionStatus.FAILED
    assert "RuntimeError" in record.error


@pytest.mark.asyncio
async def test_status_nonexistent(tmp_path):
    """查询不存在的会话返回 None"""
    store = JsonSessionStore(str(tmp_path / "sessions"))
    manager = SessionManager(store=store, agent_builder=make_agent_builder(FakeAgent("x")))
    assert await manager.get_status("ghost") is None


def test_get_session_manager_is_singleton():
    """
    会话管理器必须是进程级单例。

    SessionManager._sessions 持有后台任务的 asyncio.Task 句柄，
    （取消任务等内存态能力依赖它）。
    若每个请求新建实例，_sessions 每次都是空的，这些能力会失效。
    """
    from app.main import get_session_manager

    m1 = get_session_manager()
    m2 = get_session_manager()
    assert m1 is m2, "get_session_manager 应返回同一实例（lru_cache 单例）"

    # 单例实例的 _sessions 字典也应一致
    assert m1._sessions is m2._sessions
