"""Small real-only kernel acceptance. No canned model answers or tool fixtures."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage

from travel_agent.cli import _trace_redact
from travel_agent.config import Settings
from travel_agent.graph.builder import build_graph
from travel_agent.persistence import Database

CASES = {
    "basic": [
        "你好。",
        "最近有空想出去走走，你帮我看看去哪里合适。",
    ],
    "local_edit": [
        "酒店换一家，选交通方便、预算合适的就行，你帮我决定，更新攻略，其他都保留。",
        "现在不用再改，简单说说每天分别安排了什么。",
    ],
    "commit_edit": [
        "继续完成刚才的住宿调整并更新攻略，已有资料够用的话不用再搜，景点和交通都保留。",
        "现在先不改了，说说哪一天走得最多，依据是什么。",
    ],
    "report_edit": [
        "住宿换一版，优先安静、评分高的，更新攻略，景点和交通都保持。",
        "先不用再改，解释一下这趟哪一天最累，为什么。",
    ],
    "compare_once": [
        "去印尼玩，日惹和巴厘岛各适合什么兴趣？先比较一下，不排每天。",
    ],
    "comparison": [
        "去印尼玩，日惹和巴厘岛各适合什么兴趣？先比较一下，不排每天。",
        "再找一个不同于这两个的方向，简单说优缺点。",
    ],
    "short_ideas": [
        "两个人香港出发，12月下旬有五天，每人八千人民币以内，喜欢暖和的海边，不想太赶。先推荐两个方向，我选好再安排。",
        "除了刚才这两个，再给一个不同的选择，简单比较一下。",
    ],
    "expand_only": [
        "除了这些，还有别的地方和走法吗？再给两种不同的，我先看看。",
    ],
    "dates": [
        "两个人香港出发，2026年12月23日至2027年1月2日，去印尼和新加坡，12天11晚，帮我安排。",
        "那就改成1月3日回来，先确认总共几天，不急着排每天。",
    ],
    "exploration": [
        "香港出发，两个人，12月23日到1月1日，想去印尼，每人一万二左右，喜欢自然和历史，别太赶。先给我三种地方组合，我选好再排每天。",
        "除了这些，还有别的地方和走法吗？再给两种不同的，我先看看。",
    ],
    "conversation": [
        "你好。",
        "最近有空想出去走走，你帮我看看去哪里合适。",
        "香港出发，两个人，12月23日到1月1日，想去印尼，每人一万二左右，喜欢自然和历史，别太赶。先给我三种地方组合，我选好再排每天。",
        "除了这些，还有别的地方和走法吗？再给两种不同的，我先看看。",
    ],
    "rail": [
        "明天两个人从广州南去长沙南，上午出发，帮我比较几趟高铁的时间和二等座价格。",
    ],
    "conflict": [
        "两个人香港出发，2026年12月23日至2027年1月2日，去印尼和新加坡，12天11晚，帮我安排。",
        "日期改成1月3日，还是先给我两个地方组合，不急着做详细安排。",
    ],
    "report": [
        "两个人从上海出发，2026年11月24日至26日去杭州，总预算6000人民币，喜欢湖边散步、园林和小馆子，节奏轻松些。帮我把这三天安排好，给我一份能保存的攻略。",
    ],
    "revision": [
        "酒店想住得更安静舒服些，换两家让我比较一下，其他安排先照旧。",
        "先不用改了，解释一下这趟哪一天最累，为什么。",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--root")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.root or ("output/kernel-smoke/" + time.strftime("%Y%m%d-%H%M%S")))
    root.mkdir(parents=True, exist_ok=True)
    prefix = args.case + "-" + time.strftime("%Y%m%d-%H%M%S")
    settings = replace(
        Settings.from_env(), tool_mode="real", require_real_model=True,
        database_path=root / "session.sqlite3", output_dir=root,
    )
    db = Database(settings.database_path)
    graph, registry, gateway = build_graph(settings, db)
    gateway.capture_io = True
    assert all("mock" not in type(t).__name__.lower() for t in registry.tools.values())
    session = {"revision": "report", "report_edit": "report", "commit_edit": "report", "local_edit": "report", "expand_only": "exploration"}.get(args.case, args.case)
    thread_id, trip_id = f"kernel:{session}", f"kernel-{session}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    turns = []
    original_run = registry.run

    def traced(name, arguments, thread):
        print("TOOL", name, flush=True)
        return original_run(name, arguments, thread)

    registry.run = traced
    print("ROOT", root.resolve(), flush=True)
    if args.resume:
        saved_state = graph.get_state(config)
        if saved_state.values:
            # Recover the real workspace, not the old overflowing message bodies.
            # All model/tool events and complete evidence remain in SQLite.
            graph.update_state(config, {
                "agent_messages": [], "planner_round": 0, "loaded_tools": [],
                "final_checks": 0, "agent_done": False, "pending_calls": [],
            }, as_node="execute_actions")
            print("RESUME real workspace and evidence; reset only transient loop messages", flush=True)
    try:
        for index, query in enumerate(CASES[args.case], 1):
            prior_calls = len(db.list_tool_executions(thread_id, limit=10000))
            prior_usage = len(gateway.attempt_trace)
            old_report = db.latest_report(trip_id)
            print("USER", query, flush=True)
            value = {
                "messages": [HumanMessage(content=query)], "user_id": "kernel",
                "trip_id": trip_id, "thread_id": thread_id, "current_query": query,
            }
            result = graph.invoke(None if args.resume and index == 1 else value, config)
            new_calls = db.list_tool_executions(thread_id, limit=10000)[prior_calls:]
            answer = result.get("assistant_response", "")
            saved = db.latest_report(trip_id)
            usage = gateway.attempt_trace[prior_usage:]
            entry = {
                "query": query, "response": answer,
                "tools": [{"name": c["tool_name"], "arguments": _trace_redact(c["payload"]["arguments"]),
                           "success": c["payload"]["result"]["success"]} for c in new_calls],
                "report_before": old_report["version"] if old_report else None,
                "report_after": saved["version"] if saved else None,
                "model_calls": len(usage),
                "input_tokens": sum(t.get("usage", {}).get("input_tokens", 0) for t in usage),
                "output_tokens": sum(t.get("usage", {}).get("output_tokens", 0) for t in usage),
            }
            if saved:
                payload = json.loads(saved["payload_json"])["report"]
                entry["report_quality"] = payload.get("quality")
                (root / f"{prefix}-turn{index}-report.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            turns.append(entry)
            print("ASSISTANT", answer, flush=True)
            print("USAGE", entry["model_calls"], entry["input_tokens"], entry["output_tokens"], flush=True)
            assert answer, "No actual user answer"
            if args.case in {"conversation", "rail", "conflict", "dates", "exploration", "expand_only", "short_ideas", "comparison", "compare_once"}:
                assert entry["report_after"] == entry["report_before"], "Unexpected report generation"
            if args.case == "report":
                assert saved, "No report was generated"
    finally:
        (root / f"{prefix}-results.json").write_text(json.dumps(_trace_redact(turns), ensure_ascii=False, indent=2), encoding="utf-8")
        (root / f"{prefix}-model-trace.json").write_text(json.dumps(_trace_redact(gateway.attempt_trace), ensure_ascii=False, indent=2), encoding="utf-8")
        gateway.close()
        registry.close()
        db.close()


if __name__ == "__main__":
    main()
