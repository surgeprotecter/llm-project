from travel_agent.graph.nodes import _extract_destination, _extract_origin, _merge_requirements
from travel_agent.models import TripState


def test_extract_chinese_numbers_and_origin_destination() -> None:
    state = _merge_requirements(TripState(), "我从香港出发去成都玩五天，两个人预算一万二千元，喜欢美食")
    assert state.origin == "香港"
    assert state.destination == "成都"
    assert state.duration_days == 5
    assert state.traveler_count == 2


def test_generic_place_request_is_not_treated_as_destination() -> None:
    query = "我从香港出发，想去一个不太热的地方玩五天"
    assert _extract_origin(query) == "香港"
    assert _extract_destination(query) is None


def test_duration_and_budget_range_are_recovered_from_follow_up() -> None:
    state = _merge_requirements(TripState(), "我要去岘港，两人，从香港出发")
    state = _merge_requirements(state, "4日游，2000-3000元")
    assert state.destination == "岘港"
    assert state.duration_days == 4
    assert state.traveler_count == 2
    assert state.budget.amount == 2500
    assert state.budget.min_amount == 2000
    assert state.budget.max_amount == 3000


def test_explicit_travel_dates_are_extracted() -> None:
    state = _merge_requirements(TripState(), "2027年3月2日到3月6日去岘港")
    assert state.travel_dates.start_date.isoformat() == "2027-03-02"
    assert state.travel_dates.end_date.isoformat() == "2027-03-06"


def test_xhs_research_prefix_does_not_become_destination() -> None:
    query = "请在小红书搜索岘港旅游攻略，重点找本地人推荐景点和避坑信息。"

    assert _extract_destination(query) == "岘港"
