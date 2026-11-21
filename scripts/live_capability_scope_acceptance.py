"""Small real-model/provider acceptance suite; never uses a user's session DB."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage

from travel_agent.cli import _trace_redact
from travel_agent.config import Settings
from travel_agent.graph.builder import build_graph
from travel_agent.persistence import Database


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="new directory for an isolated run")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = replace(Settings.from_env(), tool_mode="real", require_real_model=True,
                       model_name="deepseek-flash", summary_model_name="deepseek-flash",
                       database_path=root / "session.sqlite3", output_dir=root)
    db = Database(settings.database_path)
    graph, registry, gateway = build_graph(settings, db)
    gateway.capture_io = True
    assert registry.has("google_flights__search_flights"), "Google Flights was not connected"
    assert all("mock" not in type(tool).__name__.lower() for tool in registry.tools.values())
    evidence = []

    def turn(trip, query):
        thread = f"acceptance:{trip}"
        previous = len(gateway.attempt_trace)
        print("USER", query, flush=True)
        run = graph.invoke(
            {"messages": [HumanMessage(content=query)], "user_id": "acceptance", "trip_id": trip,
             "thread_id": thread, "current_query": query},
            {"configurable": {"thread_id": thread}, "recursion_limit": 100},
        )
        answer = run["assistant_response"]
        sample = gateway.attempt_trace[previous:]
        evidence.append({"trip": trip, "query": query, "answer": answer, "scope": run["request_scope"],
                         "pending_plan": run.get("pending_plan"), "approval": run.get("plan_approval"),
                         "unpresented": run.get("unpresented_plan"), "calls": run["turn_results"],
                         "input_tokens": sum(t.get("usage", {}).get("input_tokens") or 0 for t in sample),
                         "output_tokens": sum(t.get("usage", {}).get("output_tokens") or 0 for t in sample)})
        print("ASSISTANT", answer, flush=True)
        print("TOOLS", [(c["name"], c["success"]) for c in run["turn_results"]], flush=True)
        return run

    def ask_without_old_approval(run, trip, version):
        assert "出发" in run["assistant_response"] and "？" in run["assistant_response"]
        assert run["unpresented_plan"] and run["pending_plan"]["version"] == version
        assert run.get("plan_approval", {}).get("status") != "approved"
        assert not any(registry.has(c["name"]) for c in run["turn_results"])
        assert db.latest_report(trip) is None

    try:
        trip = "pending-revision"
        first = turn(trip, "想在岘港当地玩四天，两个人，不用管到岘港的机票。先列一下你准备怎么安排，等我确认了再做。")
        assert first["request_scope"]["scenario"] == "plan_review"
        version = first["pending_plan"]["version"]
        second = turn(trip, "改一下：机票也帮我查，12月25到月底哪天便宜就哪天出发，第三天不去会安。")
        ask_without_old_approval(second, trip, version)
        third = turn(trip, "嗯继续吧")
        ask_without_old_approval(third, trip, version)
        revised = turn(trip, "香港和深圳出发都行，按机票价格择期。")
        assert revised["pending_plan"]["version"] == version + 1 and not revised["unpresented_plan"]
        assert not revised.get("plan_approval")
        assert set(revised["request_scope"]["departure_origins"]) == {"香港", "深圳"}
        assert "会安" in revised["assistant_response"] and "25" in revised["assistant_response"]
        assert not any(registry.has(c["name"]) for c in revised["turn_results"])

        trip = "live-fares"
        quote = turn(trip, "一个人，2027年1月10日去曼谷，帮我比较从香港或者深圳飞的单程经济舱机票，只看当天和两个出发地，给最低价。")
        calls = db.list_tool_executions(f"acceptance:{trip}", limit=200)
        actual = [row["payload"] for row in calls if row["tool_name"] == "google_flights__search_flights"
                  and row["payload"]["result"]["success"] and not row["payload"]["result"].get("reused")]
        origins = {iata for row in actual for iata in row["arguments"]["origin"].split(",")}
        assert {"HKG", "SZX"} <= origins, origins
        assert "香港" in quote["assistant_response"] and "深圳" in quote["assistant_response"]
        assert db.latest_report(trip) is None and not quote.get("plan_approval")
        assert all(row["payload"]["result"]["source_type"] != "mock" for row in calls)

        chat = turn("local-what-to-see", "京都的清水寺和伏见稻荷大社分别是什么？简单介绍一下就行。")
        assert "清水寺" in chat["assistant_response"] and "稻荷" in chat["assistant_response"]
        assert not any(c["name"] not in {"read_state", "read_result"} for c in chat["turn_results"])
        print("PASS", root, flush=True)
    finally:
        (root / "turns.json").write_text(json.dumps(_trace_redact(evidence), ensure_ascii=False, indent=2))
        (root / "model_trace.json").write_text(json.dumps(_trace_redact(gateway.attempt_trace), ensure_ascii=False, indent=2, default=str))
        gateway.close()
        registry.close()
        db.close()


if __name__ == "__main__":
    main()
