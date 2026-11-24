"""Tests for the deterministic progress digest used when the model cannot answer."""
from travel_agent.graph.engine import saved_progress_reply


def _state(**overrides):
    state = {
        "semantic_summary": {},
        "report_workspace": {},
        "task_plan": [],
    }
    state.update(overrides)
    return state


def test_workspace_days_render_stops():
    state = _state(
        report_workspace={"data": {"days": [{
            "day": 1,
            "date": "2026-11-25",
            "city": "香港",
            "stops": [{
                "start_time": "09:00",
                "name": "香港大学美术博物馆",
                "duration_minutes": 45,
            }],
        }]}},
    )
    reply = saved_progress_reply(state)
    assert "香港大学美术博物馆" in reply
    assert "Day 1" in reply
    assert "2026-11-25" in reply
    assert "09:00" in reply
    assert "45 分钟" in reply


def test_summary_only_digest():
    state = _state(semantic_summary={
        "decisions": ["已确认按计划 v3 执行"],
        "findings": ["天星小轮中环-尖沙咀约 9 分钟"],
    })
    reply = saved_progress_reply(state)
    assert "已确认按计划 v3 执行" in reply
    assert "天星小轮" in reply


def test_pending_steps_listed():
    state = _state(task_plan=[
        {"id": "t1", "description": "核实港大位置", "status": "completed"},
        {"id": "t2", "description": "核实天星小轮班次", "status": "in_progress"},
    ])
    reply = saved_progress_reply(state)
    assert "核实天星小轮班次" in reply
    assert "进行中" in reply
    assert "核实港大位置" not in reply


def test_empty_state_has_no_digest():
    assert saved_progress_reply(_state()) == ""
