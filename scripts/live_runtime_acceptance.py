"""Two natural, real-service acceptance journeys with durable recovery evidence."""
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

QUERIES = {
    "short": [
        "最近想和朋友出去玩几天，轻松一点，你有什么建议？",
        "从上海出发，两个人，2026年11月24号到26号，总共6000元。想去杭州，喜欢园林、咖啡和当地小馆子。把这三天安排好，给我一份能保存的攻略。",
        "酒店换一家交通方便的，其他安排都保留，更新攻略。",
        "现在不用修改，给我说说做到哪了，报告在哪里。",
    ],
    "long": [
        "帮我们安排斯里兰卡12天旅行，从香港出发，2026年12月23日至2027年1月3日，两个人，每人预算两万人民币左右。想走环线，了解文化历史和当地生活，也想有自然景观、入门偏累的徒步和浮潜。住宿安全、有特色、性价比合适就行。机票可以中转一次，太折腾的不要。把每天怎么走、住哪里和花多少钱都安排好，给我能保存的完整攻略。",
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=QUERIES)
    p.add_argument("--root", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--interrupt-after-city", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--query", action="append", help="Natural follow-ups for targeted real regressions in the same trip")
    p.add_argument("--context-window", type=int, default=36000)
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    settings = replace(Settings.from_env(), tool_mode="real", require_real_model=True,
                       database_path=root / "session.sqlite3", output_dir=root,
                       context_token_budget=args.context_window)
    db = Database(settings.database_path)
    graph, registry, gateway = build_graph(settings, db)
    gateway.capture_io = True
    assert not any("mock" in type(t).__name__.lower() for t in registry.tools.values())
    thread, trip = f"runtime:{args.case}", f"runtime-{args.case}"
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 100}
    prefix = time.strftime("%Y%m%d-%H%M%S")
    outcomes = []

    def progress(kind, payload):
        if kind == "report" and args.interrupt_after_city and any(
            unit.get("completed") for unit in payload.get("research_progress", {}).values()
        ):
            print("INTERRUPT after a real completed city; progress already committed", flush=True)
            args.interrupt_after_city = False
            raise KeyboardInterrupt
        if kind in {"action", "tool"}:
            print(kind.upper(), payload.get("name"), payload.get("operation", ""),
                  "REUSED" if payload.get("reused") else payload.get("status", payload.get("success", "")), flush=True)
        elif kind == "plan":
            print("PLAN", [(t["description"], t["status"]) for t in payload["tasks"]], flush=True)
    gateway.progress_callback = progress
    queries = ["继续，不用重新确认已经决定的事情，完成后给我攻略。"] if args.resume else QUERIES[args.case][args.start:args.stop_after]
    if args.query:
        queries = args.query
    print("ROOT", root.resolve(), flush=True)
    print("CONNECTED", registry.names_for_phase("all"), flush=True)
    try:
        for index, query in enumerate(queries, args.start + 1):
            before = len(db.list_tool_executions(thread, limit=10000))
            old = db.latest_report(trip)
            usage_start = len(gateway.attempt_trace)
            print("USER", query, flush=True)
            result = graph.invoke({"messages": [HumanMessage(content=query)], "user_id": "runtime", "trip_id": trip,
                                   "thread_id": thread, "current_query": query}, config)
            saved = db.latest_report(trip)
            tools = db.list_tool_executions(thread, limit=10000)[before:]
            usage = gateway.attempt_trace[usage_start:]
            outcome = {"query": query, "answer": result.get("assistant_response"),
                       "report_before": old["version"] if old else None,
                       "report_after": saved["version"] if saved else None,
                       "plan": result.get("task_plan"), "run_status": result.get("run_status"),
                       "calls": [{"tool": t["tool_name"], "arguments": t["payload"]["arguments"],
                                  "success": t["payload"]["result"]["success"],
                                  "reused": t["payload"]["result"].get("reused", False)} for t in tools],
                       "input_tokens": sum(x.get("usage", {}).get("input_tokens", 0) for x in usage),
                       "output_tokens": sum(x.get("usage", {}).get("output_tokens", 0) for x in usage)}
            outcomes.append(outcome)
            print("ASSISTANT", outcome["answer"], flush=True)
            print("USAGE", outcome["input_tokens"], outcome["output_tokens"], flush=True)
            if saved:
                report = json.loads(saved["payload_json"])["report"]
                (root / f"{prefix}-turn{index}-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                outcome["quality"] = report.get("quality")
                print("REPORT", saved["html_path"], report.get("quality"), flush=True)
            assert outcome["answer"], "No actual user response"
    except KeyboardInterrupt:
        outcomes.append({"interrupted": True})
        print("CANCELLED; durable workspace preserved", flush=True)
    except Exception as exc:
        outcomes.append({"failed": True, "error_type": type(exc).__name__, "error": str(exc)[:1000]})
        print("FAILED; real evidence and unfinished state preserved", type(exc).__name__, flush=True)
        raise
    finally:
        state = db.runtime.load(thread)
        safe_state = {k: v for k, v in state.items() if k != "agent_messages"}
        for name, value in (("turns", outcomes), ("model_trace", gateway.attempt_trace), ("state", safe_state)):
            (root / f"{prefix}-{name}.json").write_text(json.dumps(_trace_redact(value), ensure_ascii=False, indent=2, default=str))
        gateway.close()
        registry.close()
        db.close()


if __name__ == "__main__":
    main()
