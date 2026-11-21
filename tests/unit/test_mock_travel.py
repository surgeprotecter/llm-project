from travel_agent.tools.mock_travel import PlanRouteTool, SearchFlightsTool


def test_mock_route_distributes_places_without_repeating_them() -> None:
    places = [{"name": name} for name in ("A", "B", "C", "D")]

    result = PlanRouteTool().run({"places": places, "duration_days": 3})

    scheduled = [place["name"] for day in result.data["route"] for place in day["places"]]
    assert scheduled == ["A", "B", "C", "D"]
    assert len(result.data["route"]) == 3


def test_mock_flight_prices_have_a_currency() -> None:
    result = SearchFlightsTool().run({"origin": "香港", "destination": "成都"})

    assert all(option["currency"] == "CNY" for option in result.data["options"])
