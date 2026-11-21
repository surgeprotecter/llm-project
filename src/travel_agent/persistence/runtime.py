"""Durable task state and reusable tool results, separate from graph checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta

from langchain_core.messages import message_to_dict, messages_from_dict

from travel_agent.models import ToolResult


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def fingerprint(name, arguments):
    values = {k: v for k, v in arguments.items() if k not in {"xsec_token", "cookie"}}
    return hashlib.sha256((name + canonical(values)).encode()).hexdigest()


class RuntimeStore:
    def __init__(self, database):
        self.db = database
        database.connection.executescript("""
            CREATE TABLE IF NOT EXISTS runtime_sessions (
                thread_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_results (
                result_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, task_id TEXT,
                tool_name TEXT NOT NULL, fingerprint TEXT NOT NULL,
                payload_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_runtime_result_query
                ON runtime_results(thread_id,tool_name,fingerprint,created_at);
            CREATE TABLE IF NOT EXISTS runtime_task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,
                payload_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
        """)
        database.connection.commit()

    def load(self, thread_id):
        row = self.db.connection.execute(
            "SELECT payload_json FROM runtime_sessions WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if not row:
            return {}
        result = json.loads(row[0])
        result["agent_messages"] = messages_from_dict(result.get("agent_messages", []))
        return result

    def save(self, thread_id, state):
        keys = ("task_plan", "active_goal", "root_goal", "run_status", "report_workspace", "semantic_summary",
                "loaded_tools", "loaded_skills", "trip_state", "agent_messages", "summary_version")
        keys = (*keys, "selected_results", "request_scope", "plan_approval", "pending_plan", "unpresented_plan",
                "read_progress", "repeat_reads", "action_observations")
        payload = {k: state[k] for k in keys if k in state}
        payload["agent_messages"] = [message_to_dict(m) for m in state.get("agent_messages", [])]
        # Skill bodies are read from their source files, not duplicated in SQLite.
        payload["loaded_skills"] = {k: True for k in state.get("loaded_skills", {})}
        self.db.connection.execute(
            "INSERT OR REPLACE INTO runtime_sessions VALUES (?,?,?)",
            (thread_id, canonical(payload), datetime.now(UTC).isoformat()),
        )
        self.db.connection.commit()

    def update_status(self, thread_id, status):
        state = self.load(thread_id)
        if state:
            self.save(thread_id, {**state, "run_status": status})

    def archive(self, thread_id, state):
        payload = {k: v for k, v in state.items() if k != "agent_messages"}
        self.db.connection.execute("INSERT INTO runtime_task_history(thread_id,payload_json,created_at) VALUES (?,?,?)",
                                   (thread_id, canonical(payload), datetime.now(UTC).isoformat()))
        self.db.connection.commit()

    def record(self, execution, arguments=None):
        result = execution.result
        result_id = result.result_id or execution.execution_id
        state = self.load(execution.thread_id)
        task = next((t["id"] for t in state.get("task_plan", []) if t["status"] == "in_progress"), None)
        self.db.connection.execute(
            "INSERT OR IGNORE INTO runtime_results VALUES (?,?,?,?,?,?,?)",
            (result_id, execution.thread_id, task, execution.tool_name,
             fingerprint(execution.tool_name, arguments or execution.arguments),
             canonical(execution.model_dump(mode="json")), result.fetched_at.isoformat()),
        )
        self.db.connection.commit()

    def result(self, thread_id, reference):
        reference = reference.removeprefix("result:").removeprefix("evidence:")
        row = self.db.connection.execute(
            "SELECT payload_json FROM runtime_results WHERE thread_id=? AND result_id=?",
            (thread_id, reference),
        ).fetchone()
        if row:
            return json.loads(row[0])
        # Older sessions have execution records but no runtime result index.
        row = self.db.connection.execute(
            "SELECT payload_json FROM tool_executions WHERE thread_id=? AND execution_id=?",
            (thread_id, reference),
        ).fetchone()
        if row:
            return json.loads(row[0])
        trip_ids = set()
        state = self.load(thread_id)
        trip_ids.update(state.get("trip_state", {}).get("evidence_ids", []))
        if reference in trip_ids:
            found = self.db.get_evidence([reference])
            if found:
                e = found[0]
                return {"tool_name": e.source_name, "arguments": {}, "result": {
                    "success": True, "data": e.structured_data, "summary": e.content,
                    "source_type": e.source_type, "source_name": e.source_name,
                    "fetched_at": e.fetched_at.isoformat(), "result_id": reference,
                }}
        return None

    def index(self, thread_id, *, offset=0, limit=12, tool_name=None):
        clauses, params = ["thread_id=?"], [thread_id]
        if tool_name:
            clauses.append("tool_name LIKE ?")
            params.append("%" + tool_name + "%")
        rows = self.db.connection.execute(
            "SELECT result_id,task_id,tool_name,payload_json,created_at FROM runtime_results WHERE "
            + " AND ".join(clauses) + " ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [{"result_id": r[0], "task_id": r[1], "tool": r[2],
                 "arguments": json.loads(r[3]).get("arguments", {}),
                 "success": json.loads(r[3])["result"]["success"], "fetched_at": r[4]}
                for r in rows]

    def cached(self, thread_id, name, arguments):
        rows = self.db.connection.execute(
            "SELECT payload_json FROM runtime_results WHERE thread_id=? AND tool_name=? AND fingerprint=? ORDER BY created_at DESC LIMIT 3",
            (thread_id, name, fingerprint(name, arguments)),
        ).fetchall()
        for row in rows:
            execution = json.loads(row[0])
            result = ToolResult.model_validate(execution["result"])
            if self.reusable(name, result):
                return result.model_copy(update={"result_id": result.result_id or execution["execution_id"], "reused": True})
        return None

    @staticmethod
    def reusable(name, result):
        if not result.success or result.source_type not in {"mcp", "api", "cli"}:
            return False
        now = datetime.now(UTC)
        static = (name.startswith(("open_maps__", "amap_maps__")) and "weather" not in name) or "get_feed_detail" in name
        if static:
            return now - result.fetched_at < timedelta(days=1)
        if result.valid_until:
            return result.valid_until >= now
        # Provider results without expiry are reused within the same research session.
        live_quote = any(k in name for k in ("flight", "booking", "quote", "tickets", "weather"))
        window = timedelta(minutes=30) if live_quote else timedelta(days=1)
        return now - result.fetched_at < window

    def migrate_executions(self, thread_id, normalise):
        marker = self.db.connection.execute(
            "SELECT 1 FROM runtime_results WHERE thread_id=? LIMIT 1", (thread_id,)
        ).fetchone()
        if marker:
            return
        from travel_agent.models import ToolExecution
        for row in self.db.list_tool_executions(thread_id, limit=1500):
            e = ToolExecution.model_validate(row["payload"])
            if e.result.source_type in {"api", "cli", "mcp"}:
                e.result.result_id = e.execution_id
                self.record(e, normalise(e.tool_name, copy.deepcopy(e.arguments)))
