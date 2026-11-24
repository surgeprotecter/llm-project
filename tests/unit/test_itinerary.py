"""The canonical itinerary: adapters, merge rule and the two renderers."""
from travel_agent.itinerary import (
    build_itinerary,
    from_report_payload,
    from_workspace,
    merge_proposal,
    preview_from_proposal,
    render_board,
    render_chat,
)

TRIP = {
    "destinations": ["香港"],
    "duration_days": 1,
    "traveler_count": 2,
    "budget": {"amount": 5000, "currency": "CNY"},
    "travel_dates": {"start_date": "2026-11-25", "end_date": "2026-11-25"},
}

WORKSPACE = {
    "region": {"status": "scheduled"},
    "days": [
        {
            "day": 1,
            "date": "2026-11-25",
            "city": "香港",
            "title": "港大一日游",
            "intro": "按片区安排。",
            "stops": [
                {
                    "id": "osm:node:1",
                    "name": "香港大学美术博物馆",
                    "start_time": "09:30",
                    "end_time": "10:15",
                    "duration_minutes": 45,
                    "kind": "sightseeing",
                    "area": "中西区",
                    "reason": "学校历史建筑群的一部分。",
                }
            ],
            "legs": [{"mode": "步行", "duration_minutes": 12}],
        }
    ],
    "places": [{"id": "osm:node:1", "name": "香港大学美术博物馆"}],
    "hotel_regions": [
        {
            "hotels": [
                {
                    "name": "示例酒店",
                    "area": "中环",
                    "nightly_price": 1200,
                    "nights": 1,
                    "tags": ["近地铁"],
                }
            ]
        }
    ],
    "transport": [
        {
            "options": [
                {
                    "airline": "国泰航空",
                    "flight_no": "CX390",
                    "route": "上海—香港",
                    "departure_time": "08:00",
                    "arrival_time": "10:45",
                    "price": 1880,
                }
            ]
        }
    ],
    "budget": {"items": [{"label": "机票", "amount": 3760}, {"label": "住宿", "amount": 1200}]},
    "warnings": ["酒店价格为参考价"],
}


def test_from_workspace_maps_stops_hotels_flights_budget():
    itinerary = from_workspace(WORKSPACE, TRIP)
    assert itinerary is not None
    assert itinerary.destination == "香港"
    assert itinerary.traveler_count == 2
    assert itinerary.total_budget == 5000
    assert itinerary.days[0].stops[0].slot.start == "09:30"
    assert itinerary.days[0].stops[0].refs == ["osm:node:1"]
    assert itinerary.hotels[0].price_per_night == 1200
    assert itinerary.flights[0].flight_no == "CX390"
    assert [line.label for line in itinerary.budget_lines] == ["机票", "住宿"]
    assert itinerary.warnings == ["酒店价格为参考价"]


def test_render_board_and_chat_share_the_same_stops():
    itinerary = from_workspace(WORKSPACE, TRIP)
    board = render_board(itinerary)
    chat = render_chat(itinerary)

    activity = board["days"][0]["activities"][0]
    assert activity["time"] == "09:30"
    assert activity["type"] == "游览"
    assert activity["duration"] == "45 分钟"
    assert board["travelers"] == "2 位旅行者"
    assert "| 09:30–10:15 | 香港大学美术博物馆 |" in chat
    assert "示例酒店" in chat and "CX390" in chat
    assert "酒店价格为参考价" in chat


def test_from_report_payload_reads_report_dict():
    payload = {
        "report": {
            "title": "报告版行程",
            "summary": "报告摘要",
            "days": [{"day": 1, "title": "报告第一天", "stops": [{"name": "报告景点", "kind": "dining"}]}],
            "budget": {"items": [{"label": "餐饮", "amount": 300}], "total": 3000},
        },
        "trip": {"destinations": ["东京"]},
    }
    itinerary = from_report_payload(payload, None)
    assert itinerary is not None
    assert itinerary.title == "报告版行程"
    assert itinerary.destination == "东京"
    assert itinerary.days[0].stops[0].kind == "美食"
    assert itinerary.estimated_cost == 3000


def test_merge_proposal_fills_one_day_and_keeps_multi_day_trips():
    itinerary = from_workspace(WORKSPACE, TRIP)
    merged = merge_proposal(itinerary, "1. 上午 9:00 中环出发\n2. 12:00 午餐\n3. 14:00 尖沙咀")
    assert len(merged.days[0].stops) == 3

    multi = itinerary.model_copy(update={"duration_days": 3})
    assert len(merge_proposal(multi, "1. 上午 9:00 中环出发\n2. 12:00 午餐").days[0].stops) == 1


def test_merge_proposal_leaves_a_fully_executed_day_alone():
    itinerary = from_workspace(WORKSPACE, TRIP)
    executed = itinerary.model_copy(update={"days": [itinerary.days[0].model_copy(update={"stops": [
        itinerary.days[0].stops[0],
        itinerary.days[0].stops[0].model_copy(update={"name": "第二个景点"}),
        itinerary.days[0].stops[0].model_copy(update={"name": "第三个景点"}),
    ]})]})
    merged = merge_proposal(executed, "1. 上午 9:00 中环出发\n2. 12:00 午餐")
    assert len(merged.days[0].stops) == 3


def test_build_itinerary_returns_none_for_empty_state():
    assert build_itinerary({}) is None
    assert build_itinerary({"report_workspace": {"data": {}}}, None) is None


def test_preview_from_proposal_builds_board_rows():
    trip = {"destinations": ["香港"], "duration_days": 1, "travel_dates": {"start_date": "2026-11-25"}}
    proposal = (
        "**待确认计划 v1**\n\n"
        "1. 上午 9:00 香港大学本部与美术博物馆\n"
        "2. 中午 12:30 西营盘正街用餐\n"
        "3. 下午 14:30 坚尼地城海滨\n"
        "4. 傍晚 17:30 天星小轮中环—尖沙咀\n"
    )
    itinerary = preview_from_proposal(trip, proposal)
    assert itinerary is not None
    assert itinerary.title.endswith("（待执行）")
    assert len(itinerary.days[0].stops) == 4
    assert itinerary.days[0].stops[1].kind == "美食"


def test_preview_from_proposal_ignores_research_plans():
    trip = {"destinations": ["香港"], "duration_days": 1}
    proposal = "1. 知识扫盲：确认是否需要预约。\n2. 重点核实：用地图查港大本部。\n3. 餐饮核实：查西营盘餐厅。\n"
    assert preview_from_proposal(trip, proposal) is None
