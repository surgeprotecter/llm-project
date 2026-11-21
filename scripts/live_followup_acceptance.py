"""Run a natural follow-up against a previous REAL acceptance session, with fresh tools/model."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage

from travel_agent.cli import _trace_redact
from travel_agent.config import Settings
from travel_agent.graph.builder import build_graph
from travel_agent.persistence import Database

root = Path(sys.argv[1])
query = sys.argv[2]
case = root.parent.name
settings = replace(
    Settings.from_env(),
    tool_mode="real",
    require_real_model=True,
    database_path=root / "session.sqlite3",
    output_dir=root,
)
db = Database(settings.database_path)
graph, tools, model = build_graph(settings, db)
model.capture_io = True
assert all("mock" not in type(t).__name__.lower() for t in tools.tools.values())
thread = "live:" + case
original = tools.run


def traced(name, args, thread):
    print("TOOL", name, flush=True)
    return original(name, args, thread)


tools.run = traced
stamp = time.strftime("%Y%m%d-%H%M%S")
saved = db.latest_report(case)
before = json.loads(saved["payload_json"])["report"]
before_ids = {x["execution_id"] for x in db.list_tool_executions(thread, limit=10000)}
summary = {}
try:
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "user_id": "live-adaptive",
            "trip_id": case,
            "thread_id": thread,
            "current_query": query,
        },
        {"configurable": {"thread_id": thread}, "recursion_limit": 100},
    )
    saved = db.latest_report(case)
    after = json.loads(saved["payload_json"])["report"]
    calls = [
        x
        for x in db.list_tool_executions(thread, limit=10000)
        if x["execution_id"] not in before_ids
    ]
    checks = {
        "no_clarification": not result.get("__interrupt__"),
        "quality": after["quality"]["passed"],
        "all_days": len(after["days"]) == after["trip"]["duration_days"],
        "all_cities": all(
            any(p.get("city_id") == c["id"] for p in after["places"])
            for c in after["region"]["cities"]
        ),
        "no_empty_tour_days": all(d.get("type") == "transfer" or d["stops"] for d in after["days"]),
        "hotel_coverage": all(g["hotels"] for g in after["hotel_regions"]),
        "real_models": model.call_count > 0 and model.fallback_count == 0,
        "real_tools": all(x["payload"]["result"]["source_type"] != "mock" for x in calls),
        "real_evidence": bool(after.get("sources")),
    }
    sections = set(after["revision"]["requested_sections"])
    if sections <= {"hotel_regions", "budget"}:
        checks.update(
            hotel_only_tools=all("booking" in x["tool_name"] for x in calls),
            preserved_days=after["days"] == before["days"],
            preserved_transport=after["transport"] == before["transport"],
            preserved_notes=after["notes"] == before["notes"],
            preserved_destination=after["trip"]["destinations"] == before["trip"]["destinations"],
        )
    targets = set(after["revision"].get("target_transport_ids", []))
    if targets and sections <= {"transport", "budget", "days"}:
        original_segments = {t.get("id"): t for t in before["transport"]}
        changed_segments = {
            t.get("id") for t in after["transport"] if t != original_segments.get(t.get("id"))
        }
        checks.update(
            only_requested_segments_changed=changed_segments <= targets,
            transport_edit_kept_hotels=before["hotel_regions"] == after["hotel_regions"],
            transport_edit_kept_notes=before["notes"] == after["notes"],
        )
        if len(targets) == 1:
            checks["single_leg_only_flight_tool"] = bool(calls) and all(
                "flights" in c["tool_name"] for c in calls
            )
    summary = {
        "passed": all(checks.values()),
        "checks": checks,
        "query": query,
        "response": result.get("assistant_response"),
        "quality": after["quality"],
        "report": saved["html_path"],
        "tools": [x["tool_name"] for x in calls],
        "model_calls": model.call_count,
        "input_tokens": sum(x.get("usage", {}).get("input_tokens", 0) for x in model.attempt_trace),
        "output_tokens": sum(
            x.get("usage", {}).get("output_tokens", 0) for x in model.attempt_trace
        ),
    }
    (root / f"followup-{stamp}-report.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=2)
    )
finally:
    (root / f"followup-{stamp}-model-trace.json").write_text(
        json.dumps(_trace_redact(model.attempt_trace), ensure_ascii=False, default=str, indent=2)
    )
    (root / f"followup-{stamp}-acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, default=str, indent=2)
    )
    print("RESULT", json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    db.close()
if not summary.get("passed"):
    raise SystemExit(1)
