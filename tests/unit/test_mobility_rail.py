from __future__ import annotations

import types
from datetime import date

from travel_agent.reports.mobility import _edges, _Research, country_code


def edge(origin="香港", destination="北京", origin_code="HK", destination_code="CN"):
    return {
        "id": "edge:0",
        "kind": "entry",
        "scope": "external",
        "origin": origin,
        "destination": destination,
        "from": {"name": origin, "city": origin, "country_code": origin_code},
        "to": {"name": destination, "city": destination, "country_code": destination_code},
        "date": "2026-11-26",
        "status": "unavailable",
        "options": [],
        "gaps": [],
    }


def research(results=None):
    planner = types.SimpleNamespace(provided_results=results or [], results=[])
    return _Research(planner, [])


def rail_record():
    return {
        "tool_name": "china_rail__get-tickets",
        "arguments": {"date": "2026-11-26", "fromStation": "香港", "toStation": "北京"},
        "result": {
            "success": True,
            "source_url": "https://www.12306.cn/",
            "fetched_at": "2026-11-25T10:00:00+08:00",
            "result_id": "rail-1",
            "data": {
                "trains": [
                    {
                        "start_train_code": "G382",
                        "start_time": "11:19",
                        "arrive_time": "19:30",
                        "duration": "08:11",
                        "prices": [
                            {"seat_name": "二等座", "num": "12", "price": 1248},
                            {"seat_name": "一等座", "num": "0", "price": 1997},
                        ],
                    }
                ]
            },
        },
    }


def test_country_code_aliases():
    assert country_code("香港") == "HK"
    assert country_code("Hong Kong") == "HK"
    assert country_code("澳门") == "MO"
    # Mainland city names are not in pycountry; the route layer supplies the country instead.
    assert country_code("北京") == ""
    assert country_code("") == ""


def test_rail_capable_only_china_network():
    planner = types.SimpleNamespace(results=[])
    r = _Research(planner, [])
    assert r.rail_capable(edge()) is True
    assert r.rail_capable(edge(origin_code="CN", destination_code="CN")) is True
    assert r.rail_capable(edge(origin_code="JP", destination_code="CN")) is False
    assert r.rail_capable(edge(origin_code="", destination_code="CN")) is False


def test_edges_fill_origin_country_code():
    cities = [
        {
            "id": "cn-beijing",
            "city_id": "cn-beijing",
            "name": "北京",
            "city": "北京",
            "country_code": "CN",
            "days": 3,
            "start_day": 1,
        }
    ]
    planner = types.SimpleNamespace(
        trip=types.SimpleNamespace(
            origin="香港",
            travel_dates=types.SimpleNamespace(
                start_date=date(2026, 11, 26), end_date=date(2026, 11, 28)
            ),
        ),
        geo={"cities": cities},
    )
    edges = _edges(planner, cities, True)
    entry = next(e for e in edges if e["kind"] == "entry")
    assert entry["from"]["country_code"] == "HK"
    assert entry["to"]["country_code"] == "CN"


def test_edges_inherit_country_for_mainland_origin():
    cities = [
        {
            "id": "cn-shanghai",
            "city_id": "cn-shanghai",
            "name": "上海",
            "city": "上海",
            "country_code": "CN",
            "days": 2,
            "start_day": 1,
        }
    ]
    planner = types.SimpleNamespace(
        trip=types.SimpleNamespace(
            origin="北京",
            travel_dates=types.SimpleNamespace(
                start_date=date(2026, 12, 1), end_date=date(2026, 12, 2)
            ),
        ),
        geo={"cities": cities, "country_code": "CN", "entry_mode": "rail"},
    )
    edges = _edges(planner, cities, True)
    entry = next(e for e in edges if e["kind"] == "entry")
    assert entry["from"]["country_code"] == "CN"


def test_saved_rail_reuses_12306_result():
    r = research([rail_record()])
    target = edge()
    assert r.saved_rail(target) is True
    option = target["options"][0]
    assert option["name"] == "G382"
    assert option["price"] == 1248
    assert option["departure_time"] == "11:19"
    assert option["evidence"]["validation"] == "saved_provider_response"


def test_saved_rail_skips_other_dates():
    record = rail_record()
    record["arguments"]["date"] = "2026-12-01"
    r = research([record])
    assert r.saved_rail(edge()) is False


def test_provider_gaps_rail_outside_china_network():
    r = research()
    target = edge(origin_code="JP")
    target["mode"] = "rail"
    r.rail_capable = lambda e: False  # type: ignore[assignment]
    r.provider(target)
    assert target["options"] == []
    assert any("铁路查询只覆盖" in gap for gap in target["gaps"])
