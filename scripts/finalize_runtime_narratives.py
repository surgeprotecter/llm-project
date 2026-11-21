"""Finalize saved real reports after separating route opinions from current facts.

No model, external tools or substitute data are used. Old versions and raw research
remain intact. Each new artifact records the deterministic migration explicitly.
"""
from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

from travel_agent.models import TripState
from travel_agent.persistence import Database
from travel_agent.reports.orchestration import finish, route_takeaways
from travel_agent.reports.renderer import ReportRenderer
from travel_agent.reports.workbench import schedule_signature
from travel_agent.reports.workflow import StandardTripPlanner

ROOT = Path(__file__).resolve().parents[1] / "output/runtime-upgrade"
BUSINESS = ("days", "places", "transport", "hotel_regions", "notes", "insights", "budget")


def finalize(case):
    root = ROOT / f"acceptance_{case}"
    db = Database(root / "session.sqlite3")
    try:
        thread_id, trip_id = f"runtime:{case}", f"runtime-{case}"
        state = db.runtime.load(thread_id)
        latest = db.latest_report(trip_id)
        before = json.loads(latest["payload_json"])["report"]
        if "tradeoffs" not in before["region"] and before.get("route_takeaways") == route_takeaways(before):
            print(json.dumps({"case": case, "already_finalized": latest["version"]}))
            return
        backup_path = root / f"before-narrative-v{latest['version']}.sqlite3"
        if not backup_path.exists():
            with sqlite3.connect(backup_path) as backup:
                db.connection.backup(backup)
        data = copy.deepcopy(before)
        trip = TripState.model_validate(state["trip_state"])
        request = state.get("root_goal") or state["report_workspace"].get("request", "")
        planner = StandardTripPlanner(None, None, db, trip, request, thread_id)
        planner.geo = data["region"]
        planner.sources = data.get("sources", [])
        planner.issues = data.get("research_notes", [])
        plan, _ = finish(planner, data, set(), "分离历史路线意见与当前安排；真实业务数据保持不变", before)
        assert all(plan.report[k] == before[k] for k in BUSINESS)
        assert plan.report["quality"] == before["quality"]
        assert set(before["region"].get("tradeoffs", [])) <= set(plan.report["region"]["planning_notes"])
        trip.report_version = int(latest["version"]) + 1
        plan.report["trip"] = trip.model_dump(mode="json")
        db.save_trip(trip)
        markdown, html = ReportRenderer(db, root).render(plan)
        workspace = state["report_workspace"]
        workspace["data"] = plan.report
        workspace["schedule_signature"] = schedule_signature(plan.report["region"], plan.report, workspace)
        workspace["artifacts"].update(html_path=str(html), markdown_path=str(markdown), quality=plan.report["quality"])
        state["trip_state"] = trip.model_dump(mode="json")
        db.runtime.save(thread_id, state)
        audit = {"from_version": latest["version"], "to_version": trip.report_version,
                 "model_calls": 0, "external_calls": 0,
                 "business_sections_unchanged": {k: plan.report[k] == before[k] for k in BUSINESS},
                 "historical_notes_preserved": len(plan.report["region"]["planning_notes"]),
                 "route_takeaways": plan.report["route_takeaways"], "html": str(html)}
        db.append_event(thread_id, trip.user_id, trip_id, "report.narrative_finalized", audit)
        (root / "narrative_integrity.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
        print(json.dumps(audit, ensure_ascii=False))
    finally:
        db.close()


if __name__ == "__main__":
    for case in ("short", "long"):
        finalize(case)
