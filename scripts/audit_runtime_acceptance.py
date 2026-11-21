"""Audit saved real journeys, recovery, revisions and provider usage without new API calls."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from travel_agent.reports.orchestration import route_takeaways

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/runtime-upgrade"


def read_json(path):
    return json.loads(path.read_text())


def usage(root):
    records = {}
    for path in sorted(root.glob("*-model_trace.json")):
        for record in read_json(path):
            records[(record.get("started_at"), record.get("stage"), record.get("attempt"))] = record
    totals = defaultdict(Counter)
    local_hits = 0
    for record in records.values():
        if record.get("status") == "local_cache_hit":
            local_hits += 1
            continue
        group = totals[record.get("model", "unknown")]
        group["calls"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            group[key] += (record.get("usage") or {}).get(key, 0)
        provider = record.get("provider_usage") or {}
        group["cache_hit_tokens"] += provider.get("prompt_cache_hit_tokens", 0)
        group["cache_miss_tokens"] += provider.get("prompt_cache_miss_tokens", 0)
        group["failed_attempts"] += record.get("status") != "success" or bool(record.get("parse_error"))
    return {"by_model": dict(totals), "local_model_cache_hits": local_hits,
            "observed_tokens": sum(v["total_tokens"] for v in totals.values()),
            "note": "Provider-returned usage, includes failed/rejected debugging runs. Cancelled requests without returned usage may still be billed."}


def audit(case):
    root = OUTPUT / f"acceptance_{case}"
    db = sqlite3.connect((root / "session.sqlite3").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    state = json.loads(db.execute("SELECT payload_json FROM runtime_sessions").fetchone()[0])
    plans = [state.get("task_plan", [])] + [json.loads(r[0]).get("task_plan", []) for r in db.execute(
        "SELECT payload_json FROM runtime_task_history ORDER BY id DESC")]
    rows = db.execute("SELECT * FROM reports ORDER BY version").fetchall()
    executions = [json.loads(r[0]) for r in db.execute("SELECT payload_json FROM tool_executions")]
    external = [e for e in executions if e["result"]["source_type"] != "internal"]
    events = [(r[0], json.loads(r[1])) for r in db.execute("SELECT event_type,payload_json FROM events")]
    checks = {"real_sources_only": bool(external) and all(e["result"]["source_type"] in {"mcp", "api", "cli"} for e in external),
              "report_exists": bool(rows), "run_completed": state.get("run_status") == "completed"}
    detail = {"case": case, "checks": checks, "tool_sources": dict(Counter(e["result"]["source_type"] for e in external)),
              "external_calls": len(external), "cache_reuses": sum(e["result"].get("reused", False) for e in external),
              "compactions": [v for k, v in events if k == "context.compacted"],
              "usage": usage(root), "run_status": state.get("run_status")}
    if rows:
        row = rows[-1]
        report = json.loads(row["payload_json"])["report"]
        days = report["days"]
        cities = report["region"]["cities"]
        hotels = report.get("hotel_regions", [])
        transfers = [t for t in report.get("transport", []) if t.get("scope") == "intercity"]
        checks.update({
            "machine_quality": report.get("quality", {}).get("passed") is True,
            "all_days": len(days) == report["trip"]["duration_days"],
            "no_research_gaps": all(not d.get("research_gap") and not d.get("timing_warning") for d in days),
            "no_unrendered_template": "{{" not in (ROOT / row["html_path"]).read_text() and "{%" not in (ROOT / row["html_path"]).read_text(),
            "real_note_bodies": bool(report.get("notes")) and all(n.get("content", "").strip() for n in report.get("notes", [])),
            "budget_range": 0 < report["budget"].get("low", 0) <= report["budget"].get("high", 0),
            "hotel_baselines": bool(hotels) and all(sum(bool(h.get("baseline")) for h in g.get("hotels", [])) == 1 for g in hotels if g.get("nights")),
            "current_route_narrative": "tradeoffs" not in report["region"]
            and report.get("route_takeaways") == route_takeaways(report)
            and report["summary"] == f"{len(cities)}个城市／区域，按实际抵离时间组织{report['trip']['duration_days']}天，吃喝与体验融入每日安排。",
            "plan_complete_with_refs": any(p and all(t["status"] == "completed" and t.get("result_refs") for t in p) for p in plans),
        })
        detail.update(html=str((ROOT / row["html_path"]).resolve()), version=row["version"], quality=report.get("quality"),
                      days=[{"day": d["day"], "city": d["city"], "type": d.get("type"),
                             "main_points": len(d.get("stops", [])), "timeline": len(d.get("timeline", [])),
                             "exception": d.get("exception_reason")} for d in days],
                      cities=[c["name"] for c in cities], hotel_groups=len(hotels), intercity_segments=len(transfers),
                      budget={k: report["budget"].get(k) for k in ("low", "high", "per_person_low", "per_person_high", "currency")})
        if case == "short":
            first = json.loads(rows[0]["payload_json"])["report"]
            checks["local_edit_preserves_unrelated"] = len(rows) >= 2 and all(first[k] == report[k] for k in ("days", "places", "transport", "notes", "insights"))
            checks["hotels_and_budget_changed"] = first["hotel_regions"] != report["hotel_regions"] and first["budget"] != report["budget"]
            turns = [t for p in sorted(root.glob("*-turns.json")) for t in read_json(p) if "query" in t]
            vague = next((t for t in turns if t["query"].startswith("最近")), None)
            status = next((t for t in reversed(turns) if t["query"].startswith("现在不用修改")), None)
            checks["vague_input_clarified_without_tools"] = bool(vague and not vague["calls"] and "？" in vague["answer"])
            checks["status_without_external_queries"] = bool(status and not any("__" in c["tool"] for c in status["calls"]) and status["report_before"] == status["report_after"])
        else:
            snapshots = [read_json(p) for p in sorted(root.glob("*-state.json"))]
            before = next((s for s in snapshots if s.get("run_status") == "cancelled"), {})
            old = before.get("report_workspace", {}).get("data", {})
            old_points = {p["id"] for p in old.get("places", [])}
            checks["recovered_completed_points"] = bool(old_points) and old_points <= {p["id"] for p in report.get("places", [])}
            checks["confirmed_choice_retained"] = bool(before.get("selected_results")) and before["selected_results"] == state.get("selected_results")
            checks["multi_city"] = len(cities) >= 3
            checks["all_cities_researched"] = all(state["report_workspace"].get("research_progress", {}).get(c["id"], {}).get("completed") for c in cities)
            checks["all_city_transfers"] = len(transfers) >= len(cities) - 1 and all(t.get("options") for t in transfers)
            checks["flash_compaction"] = any(k == "context.compacted" and v.get("model") == "deepseek-flash" for k, v in events)
            flights = [o for t in report["transport"] for o in t.get("options", []) if o.get("directions")]
            checks["selected_flight_adopted"] = bool(flights) and any(o.get("source_result_id") in {s["result_id"] for s in state.get("selected_results", [])} for o in flights)
            checks["flight_price_units"] = bool(flights) and all(abs(o["price"] * o["quote_passengers"] - o["source_price"]) < .02 for o in flights)
            detail["before_interruption"] = {"points": len(old_points), "selected_results": before.get("selected_results")}
            turns = [t for p in sorted(root.glob("*-turns.json")) for t in read_json(p) if "query" in t]
            exports = [t for t in turns if "重新导出一版文件" in t["query"]]
            export = exports[-1] if exports else None
            checks["reexport_actually_wrote_file"] = bool(export and export["report_after"] > export["report_before"])
            checks["reexport_without_new_search"] = bool(export) and not any("__" in call["tool"] and not call.get("reused") for call in export["calls"])
            if export and flights:
                baseline = next((o for o in flights if o.get("baseline")), flights[0])
                answer = export["answer"]
                checks["receipt_prices_and_nights"] = all(piece in answer for piece in (
                    f"{baseline['per_person_price']:,.2f}", f"{baseline['group_estimated_price']:,.2f}",
                    f"共 {sum(g['nights'] for g in hotels)} 晚"))
            integrity = root / "final_integrity.json"
            checks["final_data_integrity"] = integrity.exists() and all(read_json(integrity).values())
        layout_path = root / "qa-layout.json"
        checks["browser_layout"] = layout_path.exists() and all(v["passed"] for v in read_json(layout_path).values())
        if layout_path.exists():
            source = read_json(layout_path).get("report_file", {})
            checks["qa_matches_latest_report"] = source.get("path") == str((ROOT / row["html_path"]).resolve()) and source.get("sha256") == hashlib.sha256((ROOT / row["html_path"]).read_bytes()).hexdigest()
    detail["passed"] = bool(rows) and all(checks.values())
    (root / "acceptance.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    db.close()
    return detail


if __name__ == "__main__":
    results = [audit(case) for case in ("short", "long")]
    path = OUTPUT / "acceptance.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for result in results:
        print(json.dumps({k: result[k] for k in ("case", "passed", "checks", "usage")}, ensure_ascii=False))
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)
