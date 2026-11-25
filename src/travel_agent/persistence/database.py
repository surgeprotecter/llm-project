from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from travel_agent.models import EvidenceItem, MemoryItem, ToolExecution, TripState


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    """Small SQLite repository used by the local MVP."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.initialize()
        from travel_agent.persistence.runtime import RuntimeStore
        self.runtime = RuntimeStore(self)

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS trips (
                trip_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                trip_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_thread ON events(thread_id, event_id);
            CREATE TABLE IF NOT EXISTS memory_items (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                trip_id TEXT,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_items(user_id, trip_id, is_active);
            CREATE TABLE IF NOT EXISTS evidence_items (
                evidence_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                is_stale INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tool_executions (
                execution_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trip_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                markdown_path TEXT NOT NULL,
                html_path TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS context_manifests (
                manifest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS report_model_cache (
                cache_key TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get_report_model_cache(self, cache_key: str) -> dict | None:
        row=self.connection.execute("SELECT payload_json,created_at FROM report_model_cache WHERE cache_key=?",(cache_key,)).fetchone()
        if not row:
            return None
        from datetime import timedelta
        if datetime.now(UTC)-datetime.fromisoformat(row["created_at"])>timedelta(days=7):
            return None
        return {"value":json.loads(row["payload_json"]),"created_at":row["created_at"]}

    def save_report_model_cache(self, cache_key: str, stage: str, value: dict) -> None:
        self.connection.execute("INSERT OR REPLACE INTO report_model_cache(cache_key,stage,payload_json,created_at) VALUES (?,?,?,?)",(cache_key,stage,json.dumps(value,ensure_ascii=False),_utc_now()))
        self.connection.commit()

    def get_trip(self, trip_id: str) -> TripState | None:
        row = self.connection.execute("SELECT state_json FROM trips WHERE trip_id = ?", (trip_id,)).fetchone()
        if not row:
            return None
        return TripState.model_validate(json.loads(row["state_json"]))

    def save_trip(self, state: TripState) -> None:
        self.connection.execute(
            """
            INSERT INTO trips(trip_id, user_id, state_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(trip_id) DO UPDATE SET
                user_id=excluded.user_id,
                state_json=excluded.state_json,
                updated_at=excluded.updated_at
            """,
            (state.trip_id, state.user_id, state.model_dump_json(), _utc_now()),
        )
        self.connection.commit()

    def append_event(
        self,
        thread_id: str,
        user_id: str,
        trip_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO events(thread_id, user_id, trip_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (thread_id, user_id, trip_id, event_type, json.dumps(payload, ensure_ascii=False), _utc_now()),
        )
        self.connection.commit()

    def list_events(self, thread_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE thread_id = ? ORDER BY event_id DESC LIMIT ?",
            (thread_id, limit),
        ).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in reversed(rows)]

    def save_memory(self, item: MemoryItem) -> None:
        self.connection.execute(
            """
            INSERT INTO memory_items(memory_id, user_id, trip_id, kind, content, payload_json, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                content=excluded.content,
                payload_json=excluded.payload_json,
                is_active=excluded.is_active
            """,
            (
                item.memory_id,
                item.user_id,
                item.trip_id,
                item.kind,
                item.content,
                item.model_dump_json(),
                int(item.is_active),
                item.created_at.isoformat(),
            ),
        )
        self.connection.commit()

    def list_memories(
        self, user_id: str, trip_id: str | None = None, *, include_inactive: bool = False
    ) -> list[MemoryItem]:
        active = "" if include_inactive else " AND is_active = 1"
        rows = self.connection.execute(
            f"""
            SELECT payload_json FROM memory_items
            WHERE user_id = ?{active} AND (trip_id IS NULL OR trip_id = ?)
            ORDER BY created_at DESC
            """,
            (user_id, trip_id),
        ).fetchall()
        return [MemoryItem.model_validate(json.loads(row["payload_json"])) for row in rows]

    def list_all_memories(self, user_id: str, *, include_inactive: bool = False) -> list[MemoryItem]:
        active = "" if include_inactive else " AND is_active = 1"
        rows = self.connection.execute(
            f"SELECT payload_json FROM memory_items WHERE user_id = ?{active} ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [MemoryItem.model_validate(json.loads(row["payload_json"])) for row in rows]

    def deactivate_memory(self, memory_id: str, user_id: str | None = None) -> bool:
        sql = "UPDATE memory_items SET is_active = 0 WHERE memory_id = ?"
        args: list[object] = [memory_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        cursor = self.connection.execute(sql, args)
        self.connection.commit()
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: str, user_id: str | None = None) -> bool:
        sql = "DELETE FROM memory_items WHERE memory_id = ?"
        args: list[object] = [memory_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        cursor = self.connection.execute(sql, args)
        self.connection.commit()
        return cursor.rowcount > 0

    def save_evidence(self, item: EvidenceItem) -> None:
        self.connection.execute(
            """
            INSERT INTO evidence_items(evidence_id, source_name, payload_json, fetched_at, is_stale)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(evidence_id) DO UPDATE SET
                payload_json=excluded.payload_json,
                fetched_at=excluded.fetched_at,
                is_stale=excluded.is_stale
            """,
            (
                item.evidence_id,
                item.source_name,
                item.model_dump_json(),
                item.fetched_at.isoformat(),
                int(item.is_stale),
            ),
        )
        self.connection.commit()

    def get_evidence(self, evidence_ids: list[str]) -> list[EvidenceItem]:
        if not evidence_ids:
            return []
        placeholders = ",".join("?" for _ in evidence_ids)
        rows = self.connection.execute(
            f"SELECT payload_json FROM evidence_items WHERE evidence_id IN ({placeholders})",
            evidence_ids,
        ).fetchall()
        records = {e.evidence_id: e for row in rows for e in [EvidenceItem.model_validate(json.loads(row["payload_json"]))]}
        return [records[key] for key in dict.fromkeys(evidence_ids) if key in records]

    def save_tool_execution(self, execution: ToolExecution) -> None:
        if execution.result.result_id is None:
            execution.result.result_id = execution.execution_id
        self.connection.execute(
            """
            INSERT INTO tool_executions(execution_id, thread_id, tool_name, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                execution.execution_id,
                execution.thread_id,
                execution.tool_name,
                execution.model_dump_json(),
                _utc_now(),
            ),
        )
        self.connection.commit()

        if execution.result.source_type in {"mcp", "api", "cli"} or execution.tool_name in {
            "report_step", "itinerary_step", "export_report", "deliver_reply",
        }:
            self.runtime.record(execution)

    def list_tool_executions(self, thread_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Return tool executions in chronological order for a trace view."""
        rows = self.connection.execute(
            "SELECT * FROM tool_executions WHERE thread_id = ? ORDER BY rowid DESC LIMIT ?",
            (thread_id, limit),
        ).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in reversed(rows)]

    def save_context_manifest(self, thread_id: str, phase: str, payload: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO context_manifests(thread_id, phase, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (thread_id, phase, json.dumps(payload, ensure_ascii=False), _utc_now()),
        )
        self.connection.commit()

    def latest_context_manifest(self, thread_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM context_manifests WHERE thread_id = ? ORDER BY manifest_id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def save_report(
        self,
        trip_id: str,
        version: int,
        markdown_path: str,
        html_path: str,
        payload: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO reports(trip_id, version, markdown_path, html_path, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (trip_id, version, markdown_path, html_path, json.dumps(payload, ensure_ascii=False), _utc_now()),
        )
        self.connection.commit()

    def latest_report(self, trip_id: str, *, structured_only: bool = False) -> dict[str, Any] | None:
        constraint = " AND json_extract(payload_json, '$.report.chat_markdown') IS NULL" if structured_only else ""
        row = self.connection.execute(
            "SELECT * FROM reports WHERE trip_id = ?" + constraint + " ORDER BY version DESC LIMIT 1", (trip_id,)
        ).fetchone()
        return dict(row) if row else None
