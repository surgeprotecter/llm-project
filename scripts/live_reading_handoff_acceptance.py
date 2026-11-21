"""Replay actual saved evidence and resume an approved task with real models."""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from travel_agent.cli import _trace_redact
from travel_agent.config import Settings
from travel_agent.context.agent_context import dump
from travel_agent.context.semantic import compact_context
from travel_agent.graph.builder import build_graph
from travel_agent.graph.engine import RuntimeEngine
from travel_agent.models import TripState
from travel_agent.persistence import Database
from travel_agent.reports.renderer import ReportRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--thread", default="yuming:selftest-20261119-01")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env()
    path = root / "session.sqlite3"
    if path.exists():
        raise FileExistsError(f"Choose a fresh acceptance directory: {path}")
    with sqlite3.connect(settings.database_path.resolve().as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(path) as replica:
            source.backup(replica)
    settings = replace(settings, database_path=path, output_dir=root, require_real_model=True, tool_mode="real")
    db = Database(path)
    graph, tools, gateway = build_graph(settings, db)
    gateway.capture_io = True
    engine = RuntimeEngine(db, tools, gateway, ReportRenderer(db, root), settings)
    thread = args.thread
    checks = {}
    try:
        state = db.runtime.load(thread)
        trip = TripState.model_validate(state["trip_state"])
        trip_id, user = trip.trip_id, trip.user_id
        state.update(thread_id=thread, trip_id=trip_id, user_id=user,
                     current_query="继续已确认的计划，使用保存的资料完成行程，直接在聊天里给我。",
                     repeat_reads={}, read_progress={})
        assert state["plan_approval"]["status"] == "approved"
        approved_scope = state["request_scope"]
        assert engine.tool_spec("itinerary_step")
        assert engine.tool_spec("export_report")
        assert engine.action_error(state, trip, "export_report", {"operation": "render"})["error_type"] == "outside_request_scope"
        index = db.runtime.index(thread, tool_name="booking__booking_search", limit=100)
        ref = next(x["result_id"] for x in index if x["arguments"].get("destination") == "Cairo, Egypt"
                   and x["arguments"].get("maxResults") == 10)
        raw = db.runtime.result(thread, ref)["result"]["data"]["properties"]
        params = {"result_id": ref}
        first = engine.internal(state, trip, "read_result", params)
        cursor = first["next_args"]
        assert cursor == {"result_id": ref, "field": "properties", "offset": 3}
        assert len(first["data"]["properties"]) == 3 and "remaining_items" not in dump(first)
        checks["real_first_page"] = {"result_id": ref, "total": len(raw), "next": cursor}
        alias = {"section": "evidence", "target": ref}
        review = engine.internal(state, trip, "read_state", {**alias, "field": "properties", "mode": "review"})
        assert review["offset"] == 0 and review["items"][0]["name"] == raw[0]["name"]
        bad = engine.internal(state, trip, "read_result", {**params, "field": "remaining_items"})
        assert bad["success"] is False and bad["next_args"] == cursor
        checks["alias_review_and_invalid_field_recovery"] = True

        state["agent_messages"] = [
            HumanMessage(content=state["current_query"]),
            AIMessage(content="", tool_calls=[{"name": "read_result", "args": params, "id": "read_first_page"}]),
            ToolMessage(content=dump(first), name="read_result", tool_call_id="read_first_page"),
        ]
        version = state["summary_version"]
        compact_context(state, gateway, db, 0, force=True, keep_groups=0)
        restored = db.runtime.load(thread)
        assert restored["summary_version"] == version + 1
        assert restored["read_progress"][ref]["properties"]["ranges"] == [[0, 3]]
        state.update(restored)
        repeated = engine.internal(state, trip, "read_state", {**alias, "field": "properties", "offset": 0})
        assert repeated["offset"] == cursor["offset"] and "data" not in repeated
        gathered = [item["name"] for item in first["data"]["properties"]]
        gathered.extend(item["name"] for item in repeated["items"])
        cursor = repeated["next_args"]
        while cursor:
            page = engine.internal(state, trip, "read_result", cursor)
            gathered.extend(item["name"] for item in page["items"])
            cursor = page["next_args"]
        assert gathered == [item["name"] for item in raw]
        batch = engine.execute({**state, "pending_calls": [{"name": "read_state", "args": alias, "id": "repeat_alias"}]})
        assert batch["needs_progress"] is True and batch["delivery_feedback"]["decision"] == "no_progress"
        assert batch["turn_results"][-1]["new_information"] is False
        checks["exhausted_batch_routes_to_delivery"] = True
        checks["compaction_and_next_page"] = {"summary_version": restored["summary_version"], "total_read": len(gathered)}
        db.runtime.save(thread, state)
        print("PAGES PASSED", checks, flush=True)

        before = {r["execution_id"] for r in db.list_tool_executions(thread, limit=10000)}
        reports_before = db.latest_report(trip_id)

        def progress(kind, payload):
            if kind == "action":
                print("ACTION", payload.get("name"), payload.get("operation"), flush=True)
            if kind == "tool" and payload.get("status") == "running":
                raise AssertionError("This regression must resume from saved research")

        gateway.progress_callback = progress
        query = state["current_query"]
        run = graph.invoke({"messages": [HumanMessage(content=query)], "thread_id": thread,
                            "trip_id": trip_id, "user_id": user, "current_query": query},
                           {"configurable": {"thread_id": thread}, "recursion_limit": 100})
        calls = [r for r in db.list_tool_executions(thread, limit=10000) if r["execution_id"] not in before]
        answer = run.get("assistant_response", "")
        (root / "answer.md").write_text(answer)
        assert run["run_status"] == "completed", (run["run_status"], answer[:700])
        assert run["task_plan"][-1]["status"] == "completed" and run["task_plan"][-1]["completion"] == "reply"
        assert db.latest_report(trip_id) == reports_before
        assert [r["tool_name"] for r in calls] == ["deliver_reply"]
        assert engine.reference_exists(run, run["task_plan"][-1]["result_refs"][0])
        checks["resumed_chat_delivery"] = {"status": run["run_status"], "new_calls": [r["tool_name"] for r in calls],
                                            "characters": len(answer), "model_rounds": run["planner_round"]}
        print("DELIVERY PASSED", checks["resumed_chat_delivery"], flush=True)
    finally:
        (root / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
        (root / "model_trace.json").write_text(json.dumps(_trace_redact(gateway.attempt_trace), ensure_ascii=False, indent=2, default=str))
        gateway.close()
        tools.close()
        db.close()


if __name__ == "__main__":
    main()
