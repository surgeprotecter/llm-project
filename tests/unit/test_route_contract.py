"""Approved route contracts must stay satisfiable."""

from travel_agent.reports.route_contract import extract_route_contract


class FakeGateway:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def structured_extract(self, prompt, value, stage=None, output_tokens=None):
        self.calls.append((stage, value))
        return self.payload


APPROVED = {
    "version": 3,
    "proposal": "**港大周边一日游 · 计划（1人）**\n1. 上午：香港大学本部\n2. 中午：西营盘\n3. 下午：坚尼地城",
    "user_confirmation": "确认",
}


def test_day_walk_stops_are_not_stays():
    gateway = FakeGateway({
        "total_days": 1,
        "reserved": {"outbound": 0, "return": 0, "flex": 0},
        "stops": [
            {"name": "香港大学", "aliases": ["港大"], "min_days": None, "max_days": None},
            {"name": "西营盘", "aliases": [], "min_days": None, "max_days": None},
            {"name": "坚尼地城", "aliases": [], "min_days": None, "max_days": None},
        ],
    })
    contract = extract_route_contract(gateway, APPROVED, ["香港一日游"])
    assert contract["schema"] == 3
    assert contract["total_days"] == 1
    assert contract["stops"] == []


def test_city_stops_are_kept_when_the_arithmetic_fits():
    approved = {
        "version": 3,
        "proposal": "香港2天，澳门1天",
        "user_confirmation": "确认",
    }
    gateway = FakeGateway({
        "total_days": 3,
        "reserved": {"outbound": 0, "return": 0, "flex": 0},
        "stops": [
            {"name": "香港", "aliases": ["Hong Kong"], "min_days": 2, "max_days": 2},
            {"name": "澳门", "aliases": [], "min_days": 1, "max_days": 1},
        ],
    })
    contract = extract_route_contract(gateway, approved, [])
    assert [stop["name"] for stop in contract["stops"]] == ["香港", "澳门"]


def test_stop_outside_the_proposal_is_rejected():
    gateway = FakeGateway({
        "total_days": 2,
        "reserved": {"outbound": 0, "return": 0, "flex": 0},
        "stops": [{"name": "台北", "aliases": [], "min_days": None, "max_days": None}],
    })
    try:
        extract_route_contract(gateway, APPROVED, [])
    except ValueError as exc:
        assert "提案里没有" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")
