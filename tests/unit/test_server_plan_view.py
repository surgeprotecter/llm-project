"""One board projection, rendered from the canonical itinerary (no prose parsing)."""
from __future__ import annotations

import json

from travel_agent.itinerary import build_itinerary, render_board
from travel_agent.server import _plan_view

TRIP = {
    "destinations": ["香港"],
    "duration_days": 1,
    "traveler_count": 1,
    "budget": {"amount": 5000, "currency": "CNY"},
    "travel_dates": {"start_date": "2026-11-25", "end_date": "2026-11-25"},
}

WORKSPACE = {
    "data": {
        "days": [
            {
                "day": 1,
                "date": "2026-11-25",
                "title": "港大一日游",
                "intro": "按片区安排，并预留用餐时间。",
                "stops": [
                    {
                        "name": "香港大学美术博物馆",
                        "start_time": "09:30",
                        "duration_minutes": 45,
                        "kind": "sightseeing",
                        "area": "香港",
                    }
                ],
            }
        ]
    }
}


def test_chat_delivery_workspace_becomes_board_view():
    plan = _plan_view({"trip_state": TRIP, "report_workspace": WORKSPACE}, None)
    assert plan is not None
    assert plan["destination"] == "香港"
    assert plan["duration"] == "1 天"
    assert len(plan["days"]) == 1
    activity = plan["days"][0]["activities"][0]
    assert activity["title"] == "香港大学美术博物馆"
    assert activity["duration"] == "45 分钟"


def test_workspace_is_live_truth_even_when_older_report_row_exists():
    """A chat-delivery turn leaves the previous report row behind; the board
    must follow the executed workspace, not the stale saved snapshot."""
    row = {
        "payload_json": json.dumps(
            {"report": {"days": [{"day": 1, "title": "报告版本", "stops": [{"name": "报告景点"}]}]}},
            ensure_ascii=False,
        )
    }
    plan = _plan_view({"trip_state": TRIP, "report_workspace": WORKSPACE}, row)
    assert plan["days"][0]["theme"] == "港大一日游"
    assert plan["days"][0]["activities"][0]["title"] == "香港大学美术博物馆"


def test_report_payload_projects_when_workspace_is_empty():
    row = {
        "payload_json": json.dumps(
            {"report": {"days": [{"day": 1, "title": "报告版本", "stops": [{"name": "报告景点"}]}]}},
            ensure_ascii=False,
        )
    }
    plan = _plan_view({"trip_state": TRIP}, row)
    assert plan is not None
    assert plan["days"][0]["theme"] == "报告版本"
    assert plan["days"][0]["activities"][0]["title"] == "报告景点"


def test_stored_itinerary_short_circuits_adapters():
    itinerary = build_itinerary(
        {
            "trip_state": TRIP,
            "itinerary": {
                "title": "已保存行程",
                "destination": "香港",
                "days": [{"index": 1, "stops": [{"name": "已保存景点", "slot": {"start": "10:00"}}]}],
            },
            "report_workspace": WORKSPACE,
        }
    )
    assert itinerary is not None
    assert render_board(itinerary)["title"] == "已保存行程"
