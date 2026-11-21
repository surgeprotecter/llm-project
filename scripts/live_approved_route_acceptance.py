"""Reproduce the approved-plan handoff and route repair with Flash and real tools."""
from __future__ import annotations

import argparse
import copy
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage

from travel_agent.cli import _trace_redact
from travel_agent.config import Settings
from travel_agent.graph.builder import build_graph
from travel_agent.graph.engine import RuntimeEngine
from travel_agent.models import TripState
from travel_agent.persistence import Database
from travel_agent.reports.renderer import ReportRenderer
from travel_agent.reports.route_contract import approved_plan_view, route_issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-db", type=Path, default=Path("travel_agent.sqlite3"))
    parser.add_argument("--thread", default="yuming:selftest-20261120-165546")
    parser.add_argument("--brief", action="store_true", help="real new two-city proposal, approval and routing test")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.brief:
        return brief(root)
    with sqlite3.connect(args.source_db.resolve().as_uri() + "?mode=ro", uri=True) as source:
        row = source.execute("SELECT payload_json FROM runtime_sessions WHERE thread_id=?", (args.thread,)).fetchone()
        saved = json.loads(row[0])
        row = source.execute(
            "SELECT payload_json FROM tool_executions WHERE thread_id=? AND tool_name='itinerary_step' "
            "AND json_extract(payload_json,'$.arguments.operation')='route' ORDER BY rowid DESC LIMIT 1",
            (args.thread,),
        ).fetchone()
        old_call = json.loads(row[0])["arguments"]
    saved["agent_messages"] = []
    trip = TripState.model_validate(saved["trip_state"])
    trip.user_id, trip.trip_id, trip.thread_id = "acceptance", "approved-route", "acceptance:approved-route"
    saved["trip_state"] = trip.model_dump(mode="json")
    settings = replace(Settings.from_env(), tool_mode="real", require_real_model=True,
                       model_name="deepseek-flash", summary_model_name="deepseek-flash",
                       output_dir=root, database_path=root / "session.sqlite3")
    db = Database(settings.database_path)
    _, tools, gateway = build_graph(settings, db)
    engine = RuntimeEngine(db, tools, gateway, ReportRenderer(db, root), settings)
    gateway.capture_io = True
    checks = {}
    try:
        assert all("mock" not in type(t).__name__.lower() for t in tools.tools.values())
        db.runtime.save(trip.thread_id, saved)
        query = "继续"
        base = {"thread_id": trip.thread_id, "trip_id": trip.trip_id, "user_id": trip.user_id,
                "current_query": query, "messages": []}
        state = {**base, **engine.prepare(base)}
        original = saved["plan_approval"]
        assert approved_plan_view(state)["proposal"] == original["proposal"]
        engine.model(state)
        turn = next(t for t in gateway.attempt_trace if t["stage"] == "agent" and t["status"] == "success")
        program = next(m for m in turn["input_messages"] if m["content"].startswith("当前运行状态"))
        loaded = json.loads(program["content"].split("\n", 1)[1])["approved_plan"]
        assert loaded["proposal"] == original["proposal"] and loaded["user_confirmation"] == original["user_confirmation"]
        checks["exact_proposal_passed_to_live_model"] = loaded["version"]
        print("PASS full approved text in real model input", loaded["version"], flush=True)

        # Give the real workbench the exact old request/state from the failed
        # run. It tried to proceed with five cities and ten days in Dubai.
        previous = copy.deepcopy(state["report_workspace"]["data"]["region"]["cities"])
        denied = engine.internal(state, trip, "itinerary_step", {"operation": "transport", "transport_target": "intercity"})
        assert denied["error_type"] == "approved_route_mismatch", denied
        assert state["report_workspace"]["data"]["region"]["cities"] == previous
        assert all(city in denied["error"] for city in ("阿斯旺", "锡瓦", "马特鲁", "迪拜")), denied
        checks["invalid_existing_route_blocked"] = denied["issues"]
        print("PASS stored route blocked", denied["issues"], flush=True)
        corrected = engine.internal(state, trip, "itinerary_step", old_call)
        assert corrected["success"], corrected
        cities = corrected["cities"]
        assert not route_issues(state["plan_approval"]["route_contract"], state["report_workspace"]["data"]["region"]["cities"])
        by_name = {c["name"]: c["visit_days"] for c in cities}
        assert {"阿斯旺", "锡瓦", "马特鲁"} <= by_name.keys(), by_name
        assert by_name["迪拜"] in (1, 2) and sum(c["days"] for c in cities) == 24, by_name
        assert sum(c["days"] for c in cities) - sum(by_name.values()) == 4, cities
        executed = db.list_tool_executions(trip.thread_id, limit=100)
        map_calls = [t for t in executed if t["tool_name"] == "open_maps__search_places"
                     and t["payload"]["result"]["source_type"] == "api" and t["payload"]["result"]["success"]]
        assert map_calls, "no successful real geocoding calls"
        assert state["report_workspace"]["data"].get("transport") is None
        assert db.runtime.load(trip.thread_id)["plan_approval"]["route_contract"] == state["plan_approval"]["route_contract"]
        assert not db.latest_report(trip.trip_id), "the isolated test stops after route repair"
        checks["real_route_recovered"] = {"cities": by_name, "map_calls": len(map_calls),
                                          "contract": state["plan_approval"]["route_contract"]}
        print("PASS route repaired with live model and maps", by_name, flush=True)
    finally:
        (root / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
        (root / "model-trace.json").write_text(json.dumps(_trace_redact(gateway.attempt_trace), ensure_ascii=False, indent=2, default=str))
        gateway.close()
        tools.close()
        db.close()


def brief(root):
    settings = replace(Settings.from_env(), tool_mode="real", require_real_model=True,
                       model_name="deepseek-flash", summary_model_name="deepseek-flash",
                       output_dir=root, database_path=root / "session.sqlite3")
    db = Database(settings.database_path)
    graph, tools, gateway = build_graph(settings, db)
    engine = RuntimeEngine(db, tools, gateway, ReportRenderer(db, root), settings)
    gateway.capture_io = True
    trip = TripState.model_validate({"user_id": "acceptance", "trip_id": "brief-route",
        "thread_id": "acceptance:brief-route", "duration_days": 4, "destinations": ["日本福冈和熊本"],
        "travel_dates": {"start_date": "2027-02-10", "end_date": "2027-02-13"}})
    ids = {"thread_id": trip.thread_id, "trip_id": trip.trip_id, "user_id": trip.user_id}
    query = "我想安排2027年2月10日至13日的四天日本旅行，福冈两天、熊本两天，不另外留机动日，也不用查往返日本的机票。最后要HTML。先给我几行简短的执行计划，等我确认再做。"
    checks = {}
    try:
        first = graph.invoke({**ids, "messages": [HumanMessage(content=query)], "current_query": query},
                             {"configurable": {"thread_id": trip.thread_id}})
        print("PROPOSED", first["assistant_response"], flush=True)
        pending = first["pending_plan"]
        assert pending["scope"]["artifact"] == "report", pending
        confirmed = "确认这个计划，请按这版做。"
        state = {**ids, "current_query": confirmed, "messages": [*first["messages"], HumanMessage(content=confirmed)]}
        state.update(engine.prepare(state))
        assert state["plan_approval"]["status"] == "approved", state["request_scope"]
        assert approved_plan_view(state)["proposal"] == pending["proposal"]
        engine.save(state, trip)
        setup = engine.internal(state, trip, "itinerary_step", {"operation": "prepare"})
        assert setup["success"], setup
        result = engine.internal(state, trip, "itinerary_step", {"operation": "route"})
        assert result["success"], result
        cities = {c["name"]: c["visit_days"] for c in result["cities"]}
        assert cities == {"福冈": 2, "熊本": 2}, cities
        assert all(v == 0 for v in state["plan_approval"]["route_contract"]["reserved"].values())
        map_calls = [t for t in db.list_tool_executions(trip.thread_id, limit=40)
                     if t["tool_name"] == "open_maps__search_places" and t["payload"]["result"]["success"]]
        assert len(map_calls) >= 2 and all(t["payload"]["result"]["source_type"] == "api" for t in map_calls)
        checks.update(approved=state["plan_approval"]["version"], cities=cities, map_calls=len(map_calls))
        print("PASS new real proposal, approval and route", checks, flush=True)
    finally:
        (root / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
        (root / "model-trace.json").write_text(json.dumps(_trace_redact(gateway.attempt_trace), ensure_ascii=False, indent=2, default=str))
        gateway.close()
        tools.close()
        db.close()


if __name__ == "__main__":
    main()
