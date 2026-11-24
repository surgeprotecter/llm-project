"""Proposals (tables / numbered steps / period sections) become canonical stops."""
from travel_agent.itinerary import build_itinerary, render_board, stops_from_text

REPLY = """行程时刻表

| 时间 | 安排 | 交通/步行 |
|---|---|---|
| 08:40 | 港铁香港大学站 A 或 C 出口出站 | 站内步行约 10 分钟 |
| 09:00–09:45 | **香港大学美术博物馆** | 校园内步行 |
| 12:00–13:00 | 西营盘午餐：联华茶餐厅 | 站口步行 5–8 分钟 |
"""


def test_table_rows_parsed():
    stops = stops_from_text(REPLY)
    assert [stop.slot.start for stop in stops] == ["08:40", "09:00", "12:00"]
    assert stops[0].kind == "交通"
    assert stops[1].kind == "游览"
    assert stops[2].kind == "美食"
    assert "**" not in stops[1].name


def test_no_table_returns_empty():
    assert stops_from_text("没有表格，只有一段文字。") == []


PERIOD_REPLY = """行程总览

| 时段 | 安排 | 交通 |
|---|---|---|
| 上午 | 香港大学本部大楼一带 | 港铁港岛线至香港大学站 |
| 中午 | 西营盘用餐 | 步行或港岛线一站 |
| 下午 | 坚尼地城海滨长廊散步看海 | 港铁至坚尼地城 |
| 傍晚 | 港岛线回中环，天星小轮过海 | 港铁 + 天星小轮 |
| 晚间 | 尖沙咀海滨看维港夜景 | 步行 |
"""

PLAN = """目标：一天内步行+港铁覆盖港岛西与维港。

步骤
1. 上午 9:00 中环集合出发：中环街市—半山扶梯—荷李活道。
2. 12:00 中环一带用餐，建议茶餐厅。
3. 13:30 乘港铁到尖沙咀，星光大道—海港城一带步行。
"""


def test_period_rows_parsed():
    stops = stops_from_text(PERIOD_REPLY)
    assert [stop.slot.period for stop in stops] == ["上午", "中午", "下午", "傍晚", "晚间"]
    assert stops[1].kind == "美食"
    assert stops[3].kind == "交通"


def test_step_activities_from_proposal():
    stops = stops_from_text(PLAN)
    assert len(stops) == 3
    assert stops[0].slot.start == "9:00"
    assert stops[1].slot.start == "12:00"


SECTION_PLAN = """**上午：港大本部 + 美术博物馆**
- 港铁香港大学站下车，先逛本部大楼一带。
- 提前在 hku.hk 查看参观登记要求。

**中午：西营盘吃饭**
- 从港大步行或港岛线一站到西营盘，选一家茶餐厅用餐。

**下午：坚尼地城海滨**
- 散步到海滨长廊。
"""


def test_period_sections_parsed():
    stops = stops_from_text(SECTION_PLAN)
    assert [stop.slot.period for stop in stops] == ["上午", "中午", "下午"]
    assert stops[1].kind == "美食"
    assert "茶餐厅" in stops[1].note


def test_board_uses_approved_proposal_to_fill_a_sparse_workspace():
    result = {
        "trip_state": {"destinations": ["香港"], "duration_days": 1, "traveler_count": 1,
                       "travel_dates": {"start_date": "2026-11-25"},
                       "budget": {"amount": 5000, "currency": "CNY"}},
        "report_workspace": {"data": {"days": [{"day": 1, "date": "2026-11-25",
                                                "stops": [{"name": "香港大学美术博物馆"}]}]}},
        "plan_approval": {"status": "approved", "proposal": PLAN},
    }
    itinerary = build_itinerary(result)
    assert itinerary is not None
    board = render_board(itinerary)
    assert len(board["days"][0]["activities"]) == 3
    assert board["days"][0]["activities"][1]["title"].startswith("中环一带用餐")


def test_fully_executed_workspace_is_not_overwritten_by_the_proposal():
    result = {
        "trip_state": {"destinations": ["香港"], "duration_days": 1, "traveler_count": 1},
        "report_workspace": {"data": {"days": [{"day": 1, "stops": [
            {"name": "执行景点一", "start_time": "09:00"},
            {"name": "执行景点二", "start_time": "11:00"},
            {"name": "执行景点三", "start_time": "15:00"},
        ]}]}},
        "plan_approval": {"proposal": PLAN},
    }
    board = render_board(build_itinerary(result))
    assert [item["title"] for item in board["days"][0]["activities"]] == ["执行景点一", "执行景点二", "执行景点三"]
