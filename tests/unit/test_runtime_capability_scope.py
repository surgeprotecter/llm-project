from travel_agent.graph.delivery import classify_request, departure_options
from travel_agent.graph.engine import CORE_TOOLS, RuntimeEngine
from travel_agent.llm import ModelGateway
from travel_agent.models import TripState
from travel_agent.reports.renderer import ReportRenderer
from travel_agent.tools.live_http import build_live_http_tools
from travel_agent.tools.registry import ToolRegistry


def runtime(database, settings):
    registry = ToolRegistry(database)
    for tool in build_live_http_tools():
        registry.register(tool)
    return RuntimeEngine(database, registry, ModelGateway(settings),
                         ReportRenderer(database, settings.output_dir), settings)


def test_discovery_does_not_apply_execution_authorization(database, settings):
    engine = runtime(database, settings)
    state = {"request_scope": {"scenario": "progress", "artifact": "none"}, "loaded_tools": []}
    trip = TripState()
    names = ["export_report", "open_maps__search_places", "a_name_not_registered"]
    found = engine.internal(state, trip, "inspect_tools", {"names": names})
    assert found["loaded"] == state["loaded_tools"] == names[:2]
    assert found["unavailable"] == names[2:]
    assert engine.tool_spec("export_report")["function"]["parameters"]["properties"]["operation"]["enum"] == [
        "render", "reexport",
    ]
    assert engine.action_error(state, trip, names[0], {"operation": "render"})["error_type"] == "outside_request_scope"
    assert engine.action_error(state, trip, names[1], {})["error_type"] == "outside_request_scope"
    assert all(engine.action_error(state, trip, name, {}) is None for name in CORE_TOOLS - {"update_plan"})
    state["request_scope"]["scenario"] = "plan_review"
    assert engine.action_error(state, trip, names[1], {})["error_type"] == "outside_request_scope"
    state["request_scope"]["preapproval_lookup"] = "现在先查这个景点的位置，再给我计划"
    assert engine.action_error(state, trip, names[1], {}) is None
    assert engine.action_error(state, trip, "itinerary_step", {"operation": "route"})["error_type"] == "outside_request_scope"


def test_departure_validation_does_not_block_local_steps(database, settings):
    engine = runtime(database, settings)
    trip = TripState()
    scope = {"scenario": "execute", "artifact": "report", "requires_departure": True, "departure_origins": []}
    state = {"request_scope": scope, "plan_approval": {"status": "approved"}}
    for operation in ("research", "route"):
        assert engine.action_error(state, trip, "itinerary_step", {"operation": operation}) is None
    assert engine.action_error(state, trip, "itinerary_step", {
        "operation": "transport", "transport_target": "intercity",
    }) is None
    assert engine.action_error(state, trip, "itinerary_step", {"operation": "transport"})["field"] == "origin"
    assert engine.action_error(state, trip, "export_report", {"operation": "render"})["field"] == "origin"
    scope["departure_origins"] = ["香港", "深圳"]
    assert engine.action_error(state, trip, "itinerary_step", {
        "operation": "transport",
    })["error_type"] == "unresolved_departure_options"
    trip.origin = "深圳"
    assert engine.action_error(state, trip, "itinerary_step", {"operation": "transport"}) is None


def test_departure_candidates_must_come_from_user():
    result = {"departure_quote": "香港或者深圳都行", "departure_origins": ["香港", "深圳", "上海"]}
    assert departure_options(result, [{"speaker": "assistant", "text": result["departure_quote"]}], "都可以", []) == []
    assert departure_options(result, [], result["departure_quote"], []) == ["香港", "深圳"]


def test_unpresented_revision_blocks_implicit_approval_without_a_model_call(database, settings):
    engine = runtime(database, settings)
    old = {"scenario": "plan_review", "artifact": "report", "deliverable": "完整报告"}
    revised = {**old, "requires_departure": True, "departure_origins": []}
    base = {"pending_plan": {"version": 4, "scope": old}, "unpresented_plan": {"scope": revised}}
    assert classify_request(engine.gateway, "嗯继续吧", [], base) == (revised, None)
    base["plan_approval"] = {"version": 3, "scope": {**old, "scenario": "execute"}, "status": "approved"}
    assert classify_request(engine.gateway, "继续", [], base) == (revised, None)
    assert classify_request(engine.gateway, "继续", [], {"plan_approval": base["plan_approval"]}) == (
        base["plan_approval"]["scope"], "resume",
    )
