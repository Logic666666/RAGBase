"""
Tracer 单元测试

验证事件记录和摘要生成。
"""

from app.observability.tracer import Tracer


def test_record_and_summary():
    tracer = Tracer()
    tracer.record("start", "任务开始")
    tracer.record("tool_call", "search_kb(query=数据库)")
    tracer.record("tool_result", "[文档] 内容...")
    tracer.record("final_answer", "回答完毕。")

    summary = tracer.summary()

    assert summary["run_id"] != ""
    assert summary["event_count"] == 4
    assert summary["steps"] == 1  # 只有一次 tool_call
    assert len(summary["events"]) == 4


def test_step_increments_on_tool_call():
    tracer = Tracer()
    tracer.record("tool_call", "第一次调用")
    tracer.record("tool_result", "结果1")
    tracer.record("tool_call", "第二次调用")
    tracer.record("tool_result", "结果2")

    steps = [e["step"] for e in tracer.summary()["events"]]
    assert steps == [1, 1, 2, 2]


def test_long_detail_truncated():
    tracer = Tracer()
    long_text = "x" * 2000
    tracer.record("tool_result", long_text)

    detail = tracer.summary()["events"][0]["detail"]
    assert len(detail) <= 500


def test_summary_is_json_serializable():
    import json
    tracer = Tracer()
    tracer.record("start", "任务")
    summary = tracer.summary()
    # 能序列化说明结构正确
    json.dumps(summary)


def test_duration_ms_recorded():
    """每条事件应有 duration_ms 字段"""
    tracer = Tracer()
    tracer.record("start", "任务开始")
    tracer.record("tool_call", "search_kb")
    tracer.record("tool_result", "结果")
    tracer.record("final_answer", "回答")

    for e in tracer.summary()["events"]:
        assert "duration_ms" in e
        assert isinstance(e["duration_ms"], (int, float))


def test_total_duration_present():
    """summary 应包含总耗时"""
    tracer = Tracer()
    tracer.record("start", "任务开始")
    tracer.record("final_answer", "回答完毕")

    summary = tracer.summary()
    assert "total_duration_ms" in summary
    assert summary["total_duration_ms"] >= 0
