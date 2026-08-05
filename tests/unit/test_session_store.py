"""
会话存储单元测试

验证 JsonSessionStore 的 create/get/update/list 接口。
用临时目录，不污染真实数据。
"""

import time

import pytest

from app.sessions import JsonSessionStore, SessionRecord, SessionStatus


@pytest.fixture
def store(tmp_path):
    return JsonSessionStore(str(tmp_path / "sessions"))


@pytest.mark.asyncio
async def test_create_and_get(store):
    record = SessionRecord(session_id="abc123", kb="test", task="调研", max_steps=5)
    await store.create(record)

    fetched = await store.get("abc123")
    assert fetched is not None
    assert fetched.session_id == "abc123"
    assert fetched.task == "调研"
    assert fetched.status == SessionStatus.RUNNING


@pytest.mark.asyncio
async def test_get_nonexistent(store):
    assert await store.get("nonexistent") is None


@pytest.mark.asyncio
async def test_update(store):
    record = SessionRecord(session_id="abc123", kb="test", task="调研", max_steps=5)
    await store.create(record)

    # 部分更新：状态 + 结果 + transcript
    await store.update("abc123", {
        "status": SessionStatus.DONE,
        "result": "调研完成",
        "steps": 3,
        "messages": [
            {"role": "user", "content": "调研向量数据库"},
            {"role": "assistant", "content": "{"},  # 简单占位
        ],
        "trace": {"run_id": "abc123", "event_count": 5},
    })

    fetched = await store.get("abc123")
    assert fetched.status == SessionStatus.DONE
    assert fetched.result == "调研完成"
    assert fetched.steps == 3
    assert len(fetched.messages) == 2
    assert fetched.trace["event_count"] == 5


@pytest.mark.asyncio
async def test_list_order(store):
    # 显式错开创建时间，验证 list 按创建时间倒序
    r1 = SessionRecord(session_id="first", kb="test", task="任务1", max_steps=5,
                       created_at=time.time() - 10)
    r2 = SessionRecord(session_id="second", kb="test", task="任务2", max_steps=5,
                       created_at=time.time())
    await store.create(r1)
    await store.create(r2)

    records = await store.list()
    assert len(records) == 2
    assert records[0].session_id == "second"  # 最新的在前
    assert records[1].session_id == "first"


@pytest.mark.asyncio
async def test_update_nonexistent_no_error(store):
    """更新不存在的会话不应报错（静默忽略）"""
    await store.update("ghost", {"status": SessionStatus.DONE})
