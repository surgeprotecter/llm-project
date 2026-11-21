"""Natural user conversations through the production graph. Never loads fixtures.

Run one case at a time; real provider usage and per-turn tool differences are saved.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from travel_agent.cli import _trace_redact
from travel_agent.config import Settings
from travel_agent.graph.builder import build_graph
from travel_agent.persistence import Database

CASES = {
    "domestic_short": [
        "最近想出去散散心，别太折腾。",
        "我和朋友两个人，从上海出发，11月23日到25日，想去杭州，全部加起来6000元左右。喜欢湖边散步、喝咖啡，别把每天塞满。",
        "酒店想住得更舒服一点，换个评分高的，其他先照旧。",
    ],
    "international_short": [
        "年底想去海边待几天，不知道怎么安排。",
        "我俩从香港出发，12月10日到13日去岘港，每人8000人民币以内。想看海、逛老街、吃点当地的，轻松点就好。",
        "飞机可以多花点钱，帮我找更舒服、少转机的选择，酒店和玩的地方先不要改。",
    ],
    "domestic_long": [
        "上海出发，两个人，11月18日到29日有12天。想在苏州和杭州慢慢逛，两地各六天，两人预算两万元。喜欢园林、古镇、咖啡和手工店，想有休息时间。",
        "杭州的住宿换到交通方便、安静一点的地方，苏州那段和其他安排都保留。",
    ],
    "international_long": [
        "两个人，香港出发，2026年12月23日到2027年1月5日，去埃及玩14天，人均一万二起，上不封顶。喜欢历史、海滩和沙漠，也想逛咖啡馆和手工店，尽量避开人挤人。",
        "开罗只留两天，保留大埃及博物馆、吉萨金字塔群、孟菲斯遗址、萨拉丁城堡和悬空教堂。其他城市补具体安排，还要说明每一段怎么过去、花多久、多少钱。再加达哈卜合适吗？",
    ],
}


def main():
    case = sys.argv[1]
    root = Path("output/adaptive-acceptance") / case / time.strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
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
    thread_id = "live:" + case
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    run = tools.run

    def traced(name, args, thread):
        print("TOOL", name, flush=True)
        return run(name, args, thread)

    tools.run = traced
    snapshots = []
    turns = []
    checks = {}
    pending = False
    try:
        for i, query in enumerate(CASES[case], 1):
            if sum(t.get("usage", {}).get("total_tokens", 0) for t in model.attempt_trace) > 90000:
                raise RuntimeError("本次验收已达90000 token预算，停止追加模型请求。")
            before = len(db.list_tool_executions(thread_id, limit=10000))
            print("TURN", i, query, flush=True)
            value = (
                Command(resume=query)
                if pending
                else {
                    "messages": [HumanMessage(content=query)],
                    "user_id": "live-adaptive",
                    "trip_id": case,
                    "thread_id": thread_id,
                    "current_query": query,
                }
            )
            result = graph.invoke(value, config)
            pending = bool(result.get("__interrupt__"))
            calls = db.list_tool_executions(thread_id, limit=10000)
            delta = calls[before:]
            if i == 1 and case.endswith("short"):
                checks["vague_request_clarifies_without_tools"] = pending and not delta
            saved = db.latest_report(case)
            entry = {
                "turn": i,
                "query": query,
                "clarification": str(result.get("__interrupt__", "")),
                "response": result.get("assistant_response"),
                "tools": [c["tool_name"] for c in delta],
                "report": saved["html_path"] if saved else None,
            }
            if saved:
                report = json.loads(saved["payload_json"])["report"]
                snapshots.append(report)
                (root / f"turn{i}.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2)
                )
                checks[f"turn{i}_all_days"] = len(report["days"]) == report["trip"]["duration_days"]
                checks[f"turn{i}_no_empty_tour_days"] = all(
                    d.get("type") == "transfer" or bool(d["stops"]) for d in report["days"]
                )
                checks[f"turn{i}_city_point_coverage"] = all(
                    any(p.get("city_id") == c["id"] for p in report["places"])
                    for c in report["region"]["cities"]
                )
                checks[f"turn{i}_hotels_cover_stays"] = bool(report["hotel_regions"]) and all(
                    g["hotels"] for g in report["hotel_regions"]
                )
                checks[f"turn{i}_all_internal_edges"] = sum(
                    t.get("scope") == "intercity" for t in report["transport"]
                ) == (
                    max(0, len(report["region"]["cities"]) - 1)
                    + int(
                        report["region"].get("entry_mode") == "flight"
                        and report["region"]["cities"][-1]["id"]
                        != report["region"]["cities"][0]["id"]
                    )
                )
                checks[f"turn{i}_intercity_options"] = all(
                    t.get("options") for t in report["transport"] if t.get("scope") == "intercity"
                )
                checks[f"turn{i}_budget_arithmetic"] = (
                    sum(x["low"] for x in report["budget"]["items"]) == report["budget"]["low"]
                    and report["budget"]["low"] <= report["budget"]["high"]
                )
                checks[f"turn{i}_honest_quality_label"] = (
                    report["quality"]["passed"] or "草案" in report["status"]
                )
                checks[f"turn{i}_quality"] = report["quality"]["passed"]
                if len(snapshots) > 1:
                    old = snapshots[-2]
                    if case == "international_short":
                        checks["flight_only_no_unrelated_tools"] = all(
                            "flights" in c["tool_name"] for c in delta
                        )
                        checks["flight_only_hotels_preserved"] = (
                            report["hotel_regions"] == old["hotel_regions"]
                        )
                        checks["flight_only_notes_preserved"] = report["notes"] == old["notes"]
                    if case.startswith("domestic"):
                        checks["hotel_only_no_unrelated_tools"] = all(
                            "booking" in c["tool_name"] for c in delta
                        )
                        checks["hotel_only_transport_preserved"] = (
                            report["transport"] == old["transport"]
                        )
                        checks["hotel_only_days_preserved"] = report["days"] == old["days"]
            turns.append(entry)
            print("TURN_DONE", json.dumps(entry, ensure_ascii=False), flush=True)
    finally:
        all_calls = db.list_tool_executions(thread_id, limit=10000)
        checks["initial_and_revised_report_produced"] = len(snapshots) >= 2 and not pending
        summary = {
            "case": case,
            "passed": bool(checks) and all(checks.values()),
            "checks": checks,
            "turns": turns,
            "model_calls": model.call_count,
            "input_tokens": sum(
                t.get("usage", {}).get("input_tokens", 0) for t in model.attempt_trace
            ),
            "output_tokens": sum(
                t.get("usage", {}).get("output_tokens", 0) for t in model.attempt_trace
            ),
            "fallbacks": model.fallback_count,
            "tool_counts": dict(Counter(c["tool_name"] for c in all_calls)),
            "integration_errors": tools.integration_errors,
            "root": str(root),
        }
        (root / "acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        (root / "model-trace.json").write_text(
            json.dumps(
                _trace_redact(model.attempt_trace), ensure_ascii=False, default=str, indent=2
            )
        )
        (root / "tool-trace.json").write_text(
            json.dumps(_trace_redact(all_calls), ensure_ascii=False, default=str, indent=2)
        )
        db.close()
        print("RESULT", json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
