"""Tests for projecting a chat answer's timetable into the workbench board."""
from travel_agent.server import _plan_view, _reply_table_activities

REPLY = """行程时刻表

| 时间 | 安排 | 交通/步行 |
|---|---|---|
| 08:40 | 港铁香港大学站 A 或 C 出口出站 | 站内步行约 10 分钟 |
| 09:00–09:45 | **香港大学美术博物馆** | 校园内步行 |
| 12:00–13:00 | 西营盘午餐：联华茶餐厅 | 站口步行 5–8 分钟 |
"""


def test_table_rows_parsed():
    rows = _reply_table_activities(REPLY)
    assert [row["time"] for row in rows] == ["08:40", "09:00-09:45", "12:00-13:00"]
    assert rows[0]["type"] == "交通"
    assert rows[2]["type"] == "美食"
    assert rows[1]["type"] == "游览"
    assert "**" not in rows[1]["title"]


def test_no_table_returns_empty():
    assert _reply_table_activities("没有表格，只有一段文字。") == []


def test_plan_view_falls_back_to_reply_timetable():
    result = {
        "trip_state": {
            "destinations": ["香港"],
            "duration_days": 1,
            "traveler_count": 1,
            "travel_dates": {"start_date": "2026-11-25"},
            "budget": {"amount": 5000, "currency": "CNY"},
        },
        "report_workspace": {"data": {"days": [{
            "day": 1,
            "date": "2026-11-25",
            "stops": [{"name": "香港大学美术博物馆", "start_time": "09:00"}],
        }]}},
    }
    plan = _plan_view(result, None, REPLY)
    assert plan is not None
    activities = plan["days"][0]["activities"]
    assert len(activities) == 3
    assert activities[1]["title"] == "香港大学美术博物馆"

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
    rows = _reply_table_activities(PERIOD_REPLY)
    assert [row["time"] for row in rows] == ["上午", "中午", "下午", "傍晚", "晚间"]
    assert rows[1]["type"] == "美食"
    assert rows[3]["type"] == "交通"


def test_step_activities_from_proposal():
    from travel_agent.server import _step_activities

    rows = _step_activities(PLAN)
    assert len(rows) == 3
    assert rows[0]["time"] == "9:00"
    assert rows[1]["time"] == "12:00"


def test_plan_view_uses_approved_proposal_when_reply_has_no_table():
    result = {
        "trip_state": {"destinations": ["香港"], "duration_days": 1, "traveler_count": 1,
                       "travel_dates": {"start_date": "2026-11-25"},
                       "budget": {"amount": 5000, "currency": "CNY"}},
        "report_workspace": {"data": {"days": [{"day": 1, "date": "2026-11-25",
                                                "stops": [{"name": "香港大学美术博物馆"}]}]}},
        "plan_approval": {"status": "approved", "proposal": PLAN},
    }
    plan = _plan_view(result, None, "已按确认版整理。")
    assert plan is not None
    assert len(plan["days"][0]["activities"]) == 3

SECTION_PLAN = """**上午：港大本部 + 美术博物馆**
- 港铁香港大学站下车，先逛本部大楼一带。
- 提前在 hku.hk 查看参观登记要求。

**中午：西营盘吃饭**
- 从港大步行或港岛线一站到西营盘，选一家茶餐厅用餐。

**下午：坚尼地城海滨**
- 散步到海滨长廊。
"""


def test_period_sections_parsed():
    from travel_agent.server import _period_sections

    rows = _period_sections(SECTION_PLAN)
    assert [row["time"] for row in rows] == ["上午", "中午", "下午"]
    assert rows[1]["type"] == "美食"
    assert "茶餐厅" in rows[1]["desc"]
