"""Replay pagination and complete a blocked live multi-city trip with Flash."""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from travel_agent.config import Settings
from travel_agent.context.agent_context import dump
from travel_agent.context.semantic import compact_context
from travel_agent.graph.builder import build_graph
from travel_agent.graph.engine import RuntimeEngine
from travel_agent.models import TripState
from travel_agent.persistence import Database
from travel_agent.reports.delivery import delivered_report
from travel_agent.reports.renderer import ReportRenderer


def check_pages(engine, state, trip):
    state = {**state, "read_progress": {}, "repeat_reads": {}}
    ids = []
    for item in engine.store.index(state["thread_id"], tool_name="google_flights__search_flights", limit=30):
        args = item["arguments"]
        if item["success"] and args.get("origin") in ("HKG", "SZX") and args.get("destination") == "CAI":
            ids.append(item["result_id"])
    assert len(ids) >= 2
    checks = []
    for ref in ids[:2]:
        raw = engine.store.result(state["thread_id"], ref)["result"]["data"]["flights"]
        params = {"result_id": ref, "field": "flights", "offset": 0}
        first = engine.internal(state, trip, "read_result", {"result_id": ref})
        first_count = len(first["data"]["flights"])
        assert first["next_args"]["offset"] == first_count
        state["agent_messages"] = [
            HumanMessage(content="继续查看已保存的候选。"),
            AIMessage(content="", tool_calls=[{"id": f"read-{ref}", "name": "read_result", "args": params}]),
            ToolMessage(content=dump(first), tool_call_id=f"read-{ref}", name="read_result"),
        ]
        compact_context(state, engine.gateway, engine.db, 0, force=True, keep_groups=0)
        state.update(engine.store.load(state["thread_id"]))
        remaining = first_count
        while remaining < len(raw):
            call = {"id": f"next-{ref}-{remaining}", "name": "read_result", "args": params}
            state["agent_messages"] = [*state.get("agent_messages", []), AIMessage(content="", tool_calls=[call])]
            state.update(engine.execute({**state, "pending_calls": [call]}))
            page = json.loads(state["agent_messages"][-1].content)
            assert page["offset"] == remaining and page["returned"] > 0, page
            assert state["stalled_batches"] == 0 and state["turn_results"][-1]["new_information"]
            actual = [(p["price"], p["currency"]) for p in page["items"]]
            assert actual == [(p["price"], p["currency"]) for p in raw[remaining:remaining + len(actual)]]
            remaining += len(actual)
        drained = engine.internal(state, trip, "read_result", params)
        assert drained["exhausted"], drained
        review = engine.internal(state, trip, "read_result", {**params, "mode": "review"})
        assert review["offset"] == 0 and review["items"][0]["price"] == raw[0]["price"]
        assert engine.internal(state, trip, "read_result", params)["exhausted"]
        checks.append({"result": ref, "first_visible": first_count, "total": remaining,
                       "stale_offset_auto_advanced": True, "explicit_review": True})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trip", required=True)
    parser.add_argument("--user", default="yuming")
    parser.add_argument("--pages-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "session.sqlite3"
    if not path.exists():
        with sqlite3.connect(args.source.resolve().as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(path) as target:
                source.backup(target)
    settings = replace(Settings.from_env(), model_name="deepseek-flash", summary_model_name="deepseek-flash",
                       tool_mode="real", require_real_model=True, database_path=path, output_dir=root)
    db = Database(path)
    graph, tools, gateway = build_graph(settings, db)
    engine = RuntimeEngine(db, tools, gateway, ReportRenderer(db, root), settings)
    thread = f"{args.user}:{args.trip}"
    original = db.runtime.load(thread)
    trip = TripState.model_validate(original["trip_state"])
    state = {**original, "thread_id": thread, "trip_id": args.trip, "user_id": args.user}
    checks = {}
    gateway.progress_callback = lambda kind, data: print(
        kind, data.get("name"), data.get("operation"), data.get("status"), flush=True
    ) if kind in {"action", "tool"} else None
    try:
        if args.pages_only:
            try:
                checks["pagination"] = check_pages(engine, state, trip)
                print("PASS pagination", checks["pagination"], flush=True)
            finally:
                db.runtime.save(thread, original)
            return
        before = (db.latest_report(args.trip) or {}).get("version", 0)
        version = original["plan_approval"]["version"]
        query = "继续吧"
        result = graph.invoke({"messages": [HumanMessage(content=query)], "thread_id": thread,
                               "trip_id": args.trip, "user_id": args.user, "current_query": query},
                              {"configurable": {"thread_id": thread}})
        row = db.latest_report(args.trip)
        checks["complex"] = {"status": result.get("run_status"), "answer": result.get("assistant_response"),
                             "approved_version": result.get("plan_approval", {}).get("version"),
                             "report": row["html_path"] if row else None}
        assert result.get("run_status") == "completed" and delivered_report(row, before), checks["complex"]
        assert result["plan_approval"]["version"] == version
        payload = json.loads(row["payload_json"])["report"]
        checks["complex"].update(days=len(payload["days"]), places=payload["point_count"],
                                 segments=len(payload["transport"]), stays=len(payload["hotel_regions"]))
        print("PASS complex", checks["complex"], flush=True)
    finally:
        checks["usage"] = {"calls": len(gateway.attempt_trace),
                           "tokens": sum(i.get("usage", {}).get("total_tokens", 0) for i in gateway.attempt_trace),
                           "models": sorted({item.get("model", "") for item in gateway.attempt_trace})}
        (root / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
        gateway.close()
        tools.close()
        db.close()


if __name__ == "__main__":
    main()
