"""Local web bridge for the travel agent.

The CLI is terminal-only; this module exposes the same LangGraph runtime to the
single-page workbench in ``index.html`` (project root) over HTTP.

Start it with::

    uv run uvicorn travel_agent.server:app --reload --port 8000

Design notes:
- one process-wide :class:`AgentRuntimeService` reuses the graph, database and
  model gateway, and serialises turns with a lock (SQLite + single model key);
- ``real_only`` on a request temporarily switches the process to the strict
  runtime (``tool_mode="real"``, no mock fallback), mirroring ``--real-only``;
- the response is deliberately small: the frontend only needs ``reply``,
  ``plan`` and ``traces``. Report payloads stay in SQLite and on disk.
"""

from __future__ import annotations

import json
import threading
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from travel_agent.config import Settings
from travel_agent.graph import build_graph
from travel_agent.persistence import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_FILE = PROJECT_ROOT / "index.html"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = "default"
    trip_id: str = "default-trip"
    session_id: str | None = None
    real_only: bool = False


def _brief(value: Any, limit: int = 120) -> str:
    """Render tool arguments as one short line for the trace panel."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_traces(result: dict[str, Any], limit: int = 8) -> list[dict[str, str]]:
    """Turn the runtime's ``turn_results`` into UI trace rows.

    Each row is ``{tool, action, result, latency}``; the shape is fixed by
    ``appendAssistantMessage()`` in index.html.
    """
    traces: list[dict[str, str]] = []
    for item in list(result.get("turn_results") or [])[-limit:]:
        if not isinstance(item, dict):
            continue
        duration = item.get("duration_ms") or item.get("duration")
        latency = f"{duration}ms" if isinstance(duration, int | float) else ""
        traces.append(
            {
                "tool": str(item.get("name") or "tool"),
                "action": _brief(item.get("arguments")),
                "result": str(item.get("error") or item.get("issue") or ("成功" if item.get("success", True) else "失败")),
                "latency": latency,
            }
        )
    return traces


def _report_digest(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "version": row.get("version"),
        "html_path": row.get("html_path"),
        "markdown_path": row.get("markdown_path"),
        "created_at": row.get("created_at"),
    }


def _interrupt_question(result: dict[str, Any]) -> str | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    if isinstance(value, dict):
        return str(value.get("question") or value.get("message") or "")
    return str(value)


class AgentRuntimeService:
    """Thread-safe holder for the graph, database and model gateway."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings: Settings | None = None
        self._database: Database | None = None
        self._graph = None
        self._registry = None
        self._gateway = None
        self._current_real_only: bool | None = None

    def _close_current(self) -> None:
        if self._gateway is not None:
            self._gateway.close()
        if self._registry is not None:
            self._registry.close()
        if self._database is not None:
            self._database.close()
        self._graph = self._registry = self._gateway = self._database = None

    def _ensure_initialized(self, real_only: bool = False) -> None:
        if self._graph is not None and self._current_real_only == real_only:
            return
        self._close_current()

        settings = Settings.from_env()
        if real_only:
            settings = replace(settings, require_real_model=True, tool_mode="real")

        database = Database(settings.database_path)
        graph, registry, gateway = build_graph(settings, database)
        self._settings, self._database, self._graph = settings, database, graph
        self._registry, self._gateway = registry, gateway
        self._current_real_only = real_only

    def run_turn(
        self,
        message: str,
        user_id: str = "default",
        trip_id: str = "default-trip",
        session_id: str | None = None,
        real_only: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_initialized(real_only=real_only)
            assert self._graph is not None and self._database is not None

            thread_id = session_id or f"{user_id}:{trip_id}"
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

            state = self._graph.get_state(config)
            if getattr(state, "next", None):
                # A turn ended on a checkpoint boundary: resume it with the new
                # user message instead of starting a second thread.
                result = self._graph.invoke(Command(resume=message), config)
            else:
                initial_state = {
                    "messages": [HumanMessage(content=message)],
                    "user_id": user_id,
                    "trip_id": trip_id,
                    "thread_id": thread_id,
                    "current_query": message,
                }
                result = self._graph.invoke(initial_state, config)

            question = _interrupt_question(result)
            reply = question or result.get("assistant_response") or "本轮没有生成最终回复，请继续补充。"
            return {
                "reply": str(reply),
                "status": "needs_clarification" if question else "completed",
                "session_id": thread_id,
                "user_id": user_id,
                "trip_id": trip_id,
                "plan": result.get("plan"),
                "traces": build_traces(result),
                "report": _report_digest(self._database.latest_report(trip_id)),
            }

    @property
    def tool_mode(self) -> str | None:
        return self._settings.tool_mode if self._settings else None

    @property
    def real_only(self) -> bool | None:
        return self._current_real_only

    def close(self) -> None:
        with self._lock:
            self._close_current()


service = AgentRuntimeService()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    service.close()


app = FastAPI(title="Travel Agent Web Bridge", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail=f"{INDEX_FILE} not found; run from the project root.")
    return FileResponse(INDEX_FILE)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "real_only": service.real_only,
        "tool_mode": service.tool_mode,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    try:
        return service.run_turn(
            message=req.message,
            user_id=req.user_id,
            trip_id=req.trip_id,
            session_id=req.session_id,
            real_only=req.real_only,
        )
    except Exception as exc:  # noqa: BLE001 - the UI needs the failure reason
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
