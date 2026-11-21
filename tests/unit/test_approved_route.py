from travel_agent.context.agent_context import build_agent_messages, dump
from travel_agent.graph.engine import RuntimeEngine
from travel_agent.llm import ModelGateway
from travel_agent.models import TripState
from travel_agent.reports.renderer import ReportRenderer
from travel_agent.reports.route import calendar_cities, route_days
from travel_agent.reports.route_contract import approved_plan_view, route_issues
from travel_agent.tools.registry import ToolRegistry


def test_route_validation_totals_repeated_stops_and_accepts_aliases():
    contract = {"version": 2, "total_days": 8, "stops": [
        {"name": "马特鲁", "min_days": 1, "max_days": 2, "aliases": ["Marsa Matruh"]},
        {"name": "迪拜", "min_days": 1, "max_days": 2, "aliases": ["Dubai"]},
        {"name": "锡瓦", "min_days": None, "max_days": None, "aliases": ["Siwa"]},
    ]}
    cities = [
        {"name": "Marsa Matruh", "days": 2}, {"name": "Siwa", "days": 4},
        {"name": "Dubai", "days": 1}, {"name": "迪拜市", "days": 1},
    ]
    assert route_issues(contract, cities) == []
    # The same total is not enough if a city disappeared or a cap was exceeded.
    assert route_issues(contract, [cities[0], {"name": "Dubai", "days": 6}]) == [
        "迪拜批准的是1–2天，目前排了6天", "漏排了已批准的锡瓦",
    ]


def test_approved_proposal_is_bound_to_execution_not_the_history(database, settings):
    proposal = "路线：金泽二天、富山一天。\n" + "此处是提案正文。" * 30
    trip = TripState()
    state = {
        "thread_id": trip.thread_id, "_trip": trip, "current_query": "继续",
        "request_scope": {"scenario": "execute", "artifact": "report"},
        "plan_approval": {"version": 2, "status": "approved", "proposal": proposal, "user_confirmation": "框架可以"},
        "agent_messages": [], "semantic_summary": {"summary": "已压缩旧历史"},
    }
    messages, manifest = build_agent_messages(database, settings, state, "test policy", [], {}, [])
    assert any(dump(proposal) in msg.content for msg in messages)
    assert "approved_plan" in manifest["section_tokens"]
    state["request_scope"] = {"scenario": "answer", "artifact": "none"}
    assert approved_plan_view(state) == {}
    messages, manifest = build_agent_messages(database, settings, state, "test policy", [], {}, [])
    assert not any(dump(proposal) in msg.content for msg in messages)
    assert "approved_plan" not in manifest["section_tokens"]


def test_travel_and_buffer_days_are_not_counted_as_destination_visits():
    contract = {"version": 3, "total_days": 7,
                "reserved": {"outbound": 1, "flex": 1, "return": 1},
                "stops": [{"name": n, "aliases": [], "min_days": 2, "max_days": 2} for n in ("福冈", "熊本")]}
    cities = calendar_cities({"cities": [{"name": "福冈", "days": 1}, {"name": "熊本", "days": 3}],
                              "buffer_anchor": "熊本"}, contract, 7)
    assert not route_issues(contract, cities)
    assert [(c["days"], c.get("reserve_before"), c.get("reserve_after")) for c in cities] == [
        (4, ["outbound"], ["flex"]), (3, None, ["return"]),
    ]
    cities[0].update(id="fukuoka", start_day=1, end_day=4)
    cities[1].update(id="kumamoto", start_day=5, end_day=7)
    class Route:
        geo = {"cities": cities}
        trip = TripState.model_validate({"travel_dates": {"start_date": "2027-02-10"}})
    assert [d.get("approved_reserve_kind") for d in route_days(Route())] == [
        "outbound", None, None, "flex", None, None, "return",
    ]


def test_unsatisfied_route_stops_and_does_not_complete_approval(database, settings):
    trip = TripState()
    engine = RuntimeEngine(database, ToolRegistry(database), ModelGateway(settings),
                         ReportRenderer(database, settings.output_dir), settings)
    state = {
        "trip_id": trip.trip_id, "thread_id": trip.thread_id, "user_id": trip.user_id,
        "run_status": "running", "request_scope": {"scenario": "execute", "artifact": "report"},
        "plan_approval": {"version": 2, "status": "approved", "proposal": "去富山"},
        "turn_results": [{"error_type": "approved_route_conflict", "error": "漏排了已批准的富山"}],
    }
    result = engine.model(state)
    assert result["agent_done"] and result["run_status"] == "waiting_for_user"
    assert result["plan_approval"]["status"] == "approved"
    assert "富山" in result["assistant_response"]
    assert database.latest_report(trip.trip_id) is None
