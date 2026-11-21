"""Check continuation and HTML export with the real recorded interrupted trip."""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import HumanMessage

from travel_agent.config import Settings
from travel_agent.graph.builder import build_graph
from travel_agent.graph.engine import RuntimeEngine
from travel_agent.persistence import Database
from travel_agent.reports.delivery import delivered_report
from travel_agent.reports.renderer import ReportRenderer


class ItineraryHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)


def run_checks(root, thread, source_settings, path, legacy, report_trip, chat_copy=False):
    if not path.exists():
        with sqlite3.connect(source_settings.database_path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(path) as replica:
                source.backup(replica)
    settings = replace(source_settings, database_path=path, output_dir=root, require_real_model=True, tool_mode="real")
    database = Database(path)
    graph, registry, gateway = build_graph(settings, database)
    engine = RuntimeEngine(database, registry, gateway, ReportRenderer(database, root), settings)
    checks = {}
    try:
        gateway.progress_callback = lambda kind, data: print(
            kind, data.get("name"), data.get("operation"), data.get("status"), flush=True
        ) if kind in {"action", "tool"} else None
        original = database.runtime.load(thread)
        prior_report = database.latest_report(report_trip) if report_trip else None
        trip = (json.loads(prior_report["payload_json"])["report"]["trip"] if prior_report else original["trip_state"])
        config = {"configurable": {"thread_id": thread}}

        def send(text):
            before = {row["execution_id"] for row in database.list_tool_executions(thread, limit=10000)}
            start = len(gateway.attempt_trace)
            result = graph.invoke({"messages": [HumanMessage(content=text)], "thread_id": thread,
                                   "user_id": trip["user_id"], "trip_id": trip["trip_id"], "current_query": text}, config)
            calls = [row["tool_name"] for row in database.list_tool_executions(thread, limit=10000)
                     if row["execution_id"] not in before]
            usage = sum(x.get("usage", {}).get("total_tokens", 0) for x in gateway.attempt_trace[start:])
            return result, calls, usage

        if chat_copy:
            base = database.latest_report(trip["trip_id"])
            baseline = base["version"] if base else 0
            source = engine.export_source({"thread_id": thread}, base)
            assert source["kind"] == "reply"
            original_answer = database.runtime.result(thread, source["result_id"])["result"]["data"]["answer"]
            result, calls, tokens = send("把刚才那段聊天行程原样导出成 HTML。")
            report = database.latest_report(trip["trip_id"])
            assert calls == ["export_report"] and delivered_report(report, baseline, allow_chat=True)
            assert not delivered_report(report, baseline)
            assert Path(report["markdown_path"]).read_text(encoding="utf-8") == original_answer
            assert "聊天行程 HTML" in result["assistant_response"]
            checks["explicit_chat_export"] = {"tokens": tokens, "calls": calls, "html": report["html_path"],
                                             "same_content": True, "accepted_as_standard": False}
            print("PASS explicit chat export", checks["explicit_chat_export"], flush=True)
            return

        if prior_report:
            result, calls, tokens = send("把已经保存的完整旅行报告另存一份 HTML，原来的安排都保留。")
            new = database.latest_report(trip["trip_id"])
            assert calls == ["export_report"] and delivered_report(new, prior_report["version"]), (calls, result["assistant_response"])
            assert json.loads(new["payload_json"]) == json.loads(prior_report["payload_json"])
            assert new["html_path"] in result["assistant_response"]
            report = json.loads(new["payload_json"])["report"]
            checks["standard_reexport"] = {"tokens": tokens, "calls": calls, "html": new["html_path"],
                                          "days": len(report["days"]), "notes": len(report["notes"]),
                                          "stay_groups": len(report["hotel_regions"]),
                                          "transport_segments": len(report["transport"]),
                                          "same_content": True}
            print("PASS standard reexport", checks["standard_reexport"], flush=True)
            return

        if legacy:
            baseline = (database.latest_report(trip["trip_id"]) or {}).get("version", 0)
            result, calls, tokens = send("继续吧")
            report = database.latest_report(trip["trip_id"])
            (root / "resume-attempt.json").write_text(json.dumps({
                "status": result["run_status"], "answer": result["assistant_response"], "tokens": tokens, "calls": calls,
                "model_stages": [item["stage"] for item in gateway.attempt_trace],
            }, ensure_ascii=False, indent=2))
            assert result["run_status"] == "completed" and delivered_report(report, baseline), (calls, result["assistant_response"])
            assert "export_report" in calls and result["plan_approval"]["version"] == original["plan_approval"]["version"]
            assert "香港" in Path(report["markdown_path"]).read_text(encoding="utf-8")
            content = json.loads(report["payload_json"])["report"]
            checks["legacy_file_resume"] = {"tokens": tokens, "calls": calls, "html": report["html_path"],
                                             "original_version": result["plan_approval"]["version"], "days": len(content["days"]),
                                             "places": content["point_count"], "notes": len(content["notes"]),
                                             "stay_groups": len(content["hotel_regions"]), "transport_segments": len(content["transport"])}
            print("PASS legacy approved file", checks["legacy_file_resume"], flush=True)
            return
        ledger = [json.loads(raw) for raw, in database.connection.execute(
            "SELECT payload_json FROM runtime_task_history WHERE thread_id=? ORDER BY id", (thread,),
        )]
        saved = next(entry for entry in ledger if entry.get("plan_approval", {}).get("status") == "approved"
                     and len(entry.get("task_plan", [])) > 1 and entry["task_plan"][-2]["status"] == "completed")
        filed = next(snapshot.values for snapshot in graph.get_state_history(config, limit=300)
                     if snapshot.values.get("request_scope", {}).get("artifact") == "report"
                     and snapshot.values.get("needs_progress")
                     and snapshot.values.get("delivery_feedback", {}).get("decision") == "protocol")
        assert engine.tool_spec("export_report")
        retry = engine.repair_protocol(filed, {"agent_messages": [*filed["agent_messages"], filed["agent_messages"][-1]]})
        assert retry["protocol_repairs"] == 1 and not retry["needs_progress"]
        halted = engine.repair_protocol({**filed, **retry},
                                        {"agent_messages": [*retry["agent_messages"], retry["agent_messages"][-1]]})
        assert halted["agent_done"] and halted["run_status"] == "blocked"
        assert halted["plan_approval"]["status"] == "approved"
        checks["stalled_file_checkpoint"] = {"export_available": True, "protocol_retries": 1,
                                             "terminal_status": halted["run_status"]}
        print("PASS file-task protocol replay", flush=True)

        database.runtime.save(thread, {**original, **saved})

        result, calls, tokens = send("继续吧")
        assert result["run_status"] == "completed", result.get("assistant_response")
        assert result["plan_approval"]["version"] == saved["plan_approval"]["version"]
        assert result["task_plan"][-1]["status"] == "completed" and calls == ["deliver_reply"], calls
        answer = result["assistant_response"]
        assert engine.export_source(result, None)["kind"] == "reply"
        checks["resume"] = {"rounds": result["planner_round"], "tokens": tokens, "calls": calls,
                            "original_version": result["plan_approval"]["version"]}
        print("PASS continue", checks["resume"], flush=True)

        baseline = (database.latest_report(trip["trip_id"]) or {}).get("version", 0)
        result, calls, tokens = send("把刚才这份行程导出成 HTML。")
        report = database.latest_report(trip["trip_id"])
        assert calls == ["export_report"] and delivered_report(report, baseline, allow_chat=True), (calls, result["assistant_response"])
        assert not delivered_report(report, baseline)
        assert result["plan_approval"]["status"] == "completed"
        assert result["plan_approval"]["version"] == saved["plan_approval"]["version"]
        assert Path(report["markdown_path"]).read_text(encoding="utf-8") == answer
        parser = ItineraryHTML()
        parser.feed(Path(report["html_path"]).read_text(encoding="utf-8"))
        assert {"html", "main", "h1", "h2"} <= parser.tags
        assert report["html_path"] in result["assistant_response"]
        checks["export"] = {"tokens": tokens, "calls": calls, "html": report["html_path"], "same_markdown": True}
        print("PASS HTML", checks["export"], flush=True)
    finally:
        (root / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
        gateway.close()
        registry.close()
        database.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--thread", default="yuming:selftest-20261119-01")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--report-trip")
    parser.add_argument("--chat-copy", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env()
    if args.source_db:
        settings = replace(settings, database_path=args.source_db)
    if args.legacy:
        run_checks(root, args.thread, settings, root / "session.sqlite3", True, args.report_trip)
        return
    with TemporaryDirectory(prefix="travel-delivery-check-") as temporary:
        run_checks(root, args.thread, settings, Path(temporary) / "session.sqlite3",
                   args.legacy, args.report_trip, args.chat_copy)


if __name__ == "__main__":
    main()
