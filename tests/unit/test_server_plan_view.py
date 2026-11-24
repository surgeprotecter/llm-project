"""The workbench projection must survive chat-only deliveries."""

import json

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


def test_report_payload_days_win_over_workspace():
    row = {
        "payload_json": json.dumps(
            {"report": {"days": [{"day": 1, "title": "报告版本", "stops": [{"name": "报告景点"}]}]}},
            ensure_ascii=False,
        )
    }
    plan = _plan_view({"trip_state": TRIP, "report_workspace": WORKSPACE}, row)
    assert plan["days"][0]["theme"] == "报告版本"
    assert plan["days"][0]["activities"][0]["title"] == "报告景点"
