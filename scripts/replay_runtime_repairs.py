"""Reapply saved real actions after adapter fixes, then summarize the actual recovery."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage

from travel_agent.cli import _trace_redact
from travel_agent.config import Settings
from travel_agent.context.semantic import compact_context
from travel_agent.graph.builder import build_graph
from travel_agent.models import ToolExecution, ToolResult, TripState
from travel_agent.persistence import Database
from travel_agent.reports.workbench import report_operation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    settings = replace(Settings.from_env(), tool_mode="real", require_real_model=True,
                       database_path=root / "session.sqlite3", output_dir=root)
    db = Database(settings.database_path)
    graph, tools, gateway = build_graph(settings, db)
    gateway.capture_io = True
    prefix = time.strftime("%Y%m%d-%H%M%S") + "-repair"
    try:
        state = {**db.runtime.load("runtime:long"), "thread_id": "runtime:long", "trip_id": "runtime-long", "user_id": "runtime"}
        state["current_query"] = state["root_goal"]
        trip = TripState.model_validate(state["trip_state"])
        city = state["report_workspace"]["data"]["hotel_regions"][-1]["city"]
        hotel_args = next(json.loads(row[0])["arguments"] for row in db.connection.execute(
            "SELECT payload_json FROM tool_executions WHERE tool_name='report_step' ORDER BY rowid DESC")
            if json.loads(row[0])["arguments"].get("operation") == "hotels"
            and city in json.loads(row[0])["arguments"].get("targets", []))

        def save(workspace):
            state["report_workspace"] = workspace
            state["trip_state"] = trip.model_dump(mode="json")
            db.save_trip(trip)
            db.runtime.save("runtime:long", state)

        references = []
        for operation in ({"operation": "schedule", "request": "按已保存的交通与过夜地重建日程，保留已有选择和未受影响内容。"}, hotel_args):
            result, workspace = report_operation(db, tools, gateway, None, trip, state["report_workspace"], operation, checkpoint=save)
            assert result.get("success"), result
            execution = ToolExecution(thread_id="runtime:long", tool_name="report_step", arguments=operation,
                                      result=ToolResult(success=True, tool_name="report_step", data=result,
                                                        source_type="internal", source_name="runtime_code_repair_replay"))
            db.save_tool_execution(execution)
            references.append(execution.result.result_id)
            save(workspace)
        data = state["report_workspace"]["data"]
        last = data["hotel_regions"][-1]
        baseline = next(h for h in last["hotels"] if h["baseline"])
        assert baseline["price_is_total"], "Actual room rows still not recognized"
        assert data["days"][-1]["city"] == last["city"] and data["days"][-1]["no_hotel"]
        assert all(day.get("stops") or day.get("type") == "transfer" for day in data["days"])
        assert all([b.get("start_time", "") for b in day["timeline"]] == sorted(b.get("start_time", "") for b in day["timeline"]) for day in data["days"])
        assert not state["report_workspace"].get("active_issues"), state["report_workspace"].get("active_issues")
        notice = {"result_refs": references, "room_quote": {k: baseline.get(k) for k in ("name", "total_price", "currency", "price_is_total")},
                  "last_day_city": data["days"][-1]["city"], "day_points": [len(d.get("stops", [])) for d in data["days"]],
                  "pending": "当前工作区已修复且尚未重新导出；只需render并核对实际文件，不能再依据旧草稿重复修复。"}
        state["agent_messages"].append(HumanMessage(content="运行时真实动作重放结果（程序事实，非新增用户要求）：" + json.dumps(notice, ensure_ascii=False)))
        db.append_event("runtime:long", "runtime", "runtime-long", "runtime.repaired", notice)
        state = compact_context(state, gateway, db, 0, force=True, keep_groups=0)
        save(state["report_workspace"])
        (root / f"{prefix}-audit.json").write_text(json.dumps(notice, ensure_ascii=False, indent=2))
        print(json.dumps(notice, ensure_ascii=False))
    finally:
        (root / f"{prefix}-model_trace.json").write_text(json.dumps(_trace_redact(gateway.attempt_trace), ensure_ascii=False, indent=2, default=str))
        gateway.close()
        tools.close()
        db.close()


if __name__ == "__main__":
    main()
