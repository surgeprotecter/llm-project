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

import asyncio
import json
import sqlite3
import threading
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from travel_agent.config import Settings
from travel_agent.control import ExecutionCancelled
from travel_agent.graph import build_graph
from travel_agent.itinerary import build_itinerary, preview_from_proposal, render_board
from travel_agent.persistence import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_FILE = PROJECT_ROOT / "index.html"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = "default"
    trip_id: str = ""
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


# --------------------------------------------------------------------------- #
# Live plan projection
#
# The single-page workbench renders one flat ``plan`` object. The real runtime
# keeps its state in ``trip_state`` plus an (optional) rendered report payload,
# so the bridge projects those into the schema ``renderWorkbench()`` expects.
# Everything is defensive: a lookup-only turn still gets a usable shell, and
# unknown record shapes are skipped instead of breaking the panel.
# --------------------------------------------------------------------------- #

def _report_payload(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    raw = row.get("payload_json")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _plan_view(
    result: dict[str, Any],
    report_row: dict[str, Any] | None = None,
    reply: str = "",
) -> dict[str, Any] | None:
    """Project the canonical itinerary into the workbench ``plan`` schema.

    ``reply`` is accepted for call compatibility only: the board renders from
    :class:`travel_agent.itinerary.Itinerary` and never parses chat prose.
    """
    itinerary = build_itinerary(result, _report_payload(report_row))
    if itinerary is None:
        approval = result.get("plan_approval") if isinstance(result.get("plan_approval"), dict) else {}
        pending = result.get("pending_plan") if isinstance(result.get("pending_plan"), dict) else {}
        proposal = str(approval.get("proposal") or pending.get("proposal") or "")
        itinerary = preview_from_proposal(result.get("trip_state"), proposal)
    if itinerary is None:
        return None
    return render_board(itinerary)


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
        # thread_id -> last report row this session produced; keeps the panel
        # honest instead of showing whichever report the trip rendered last.
        self._session_reports: dict[str, dict[str, Any]] = {}
        # Only one turn runs at a time (single model key, single SQLite file), so
        # remembering which thread it belongs to keeps /api/cancel from stopping
        # someone else's queued request.
        self._active_thread: str | None = None
        self._cancel_requested = False

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
            before = self._database.latest_report(trip_id)
            self._active_thread = thread_id
            self._cancel_requested = False
            try:
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
            except (KeyboardInterrupt, ExecutionCancelled):
                snapshot = self._graph.get_state(config)
                values = dict(getattr(snapshot, "values", None) or {})
                saved = self._session_reports.get(thread_id)
                return {
                    "reply": "已取消本轮执行。已完成的步骤和查询结果都已保存，可以继续提问，或换一种说法。",
                    "status": "cancelled",
                    "session_id": thread_id,
                    "user_id": user_id,
                    "trip_id": trip_id,
                    "plan": _plan_view(values, saved, ""),
                    "traces": build_traces(values),
                    "report": None,
                }
            finally:
                self._active_thread = None

            after = self._database.latest_report(trip_id)
            report_row = None
            if after and (before is None or after.get("version") != before.get("version")):
                report_row = after
            else:
                report_row = self._session_reports.get(thread_id)
            if report_row is not None:
                self._session_reports[thread_id] = report_row

            question = _interrupt_question(result)
            reply = question or result.get("assistant_response") or "本轮没有生成最终回复，请继续补充。"
            return {
                "reply": str(reply),
                "status": "needs_clarification" if question else "completed",
                "session_id": thread_id,
                "user_id": user_id,
                "trip_id": trip_id,
                "plan": _plan_view(result, report_row, str(reply)),
                "traces": build_traces(result),
                "report": _report_digest(report_row),
            }

    def cancel(self, session_id: str) -> bool:
        """Cooperatively stop the turn running for this session.

        Must be callable while ``run_turn`` holds the lock, so it never takes it.
        """
        thread_id = (session_id or "").strip()
        if not thread_id or thread_id != self._active_thread:
            return False
        self._cancel_requested = True
        if self._graph is not None:
            self._graph.gateway.cancel()
            self._graph.registry.cancel()
        return True

    @property
    def settings(self) -> Settings | None:
        return self._settings

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
    settings = service.settings or Settings.from_env()
    real_only = service.real_only
    return {
        "status": "ok",
        "real_only": bool(real_only) if real_only is not None else settings.strict_real,
        "tool_mode": settings.tool_mode,
        "model": settings.model_name,
    }


@app.get("/api/report")
def report_file(path: str) -> FileResponse:
    """Serve a rendered HTML/Markdown report that lives under the project root."""
    root = PROJECT_ROOT.resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    try:
        resolved = target.resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"invalid report path: {exc}") from exc
    if root not in resolved.parents or resolved.suffix.lower() not in {".html", ".md"}:
        raise HTTPException(status_code=404, detail="report not found")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(resolved)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


_STAGE_LABELS = {
    "agent.request_scope": "理解请求边界",
    "agent.plan_decision": "规划下一步动作",
    "agent.result_assessed": "评估工具结果",
    "agent.delivery_decision": "判断交付条件",
    "agent.turn_finished": "收尾本轮",
    "context.compacted": "压缩上下文以腾出预算",
    "model.usage": "模型推理完成",
    "report.generated": "生成报告",
    "trip.requirements_updated": "更新行程需求",
}


def _thread_activity(thread_id: str) -> dict[str, Any]:
    """Read-only snapshot of what the runtime already did for this thread.

    The turn itself runs in a worker thread and writes `model.usage` events and
    `tool_executions` rows as it goes, so polling those two counts is enough to
    tell the browser "still working, currently on X" without touching the graph.
    """
    database = service.settings.database_path if service.settings else Settings.from_env().database_path
    try:
        conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return {}
    try:
        model_calls = int(
            conn.execute(
                "SELECT COUNT(*) FROM events WHERE thread_id = ? AND event_type = 'model.usage'",
                (thread_id,),
            ).fetchone()[0]
        )
        tools = int(conn.execute("SELECT COUNT(*) FROM tool_executions WHERE thread_id = ?", (thread_id,)).fetchone()[0])
        last = conn.execute(
            "SELECT tool_name FROM tool_executions WHERE thread_id = ? ORDER BY rowid DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        latest_event = conn.execute(
            "SELECT event_type FROM events WHERE thread_id = ? ORDER BY rowid DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {
        "model_calls": model_calls,
        "tools": tools,
        "last_tool": last[0] if last else None,
        "stage": _STAGE_LABELS.get(latest_event[0]) if latest_event else None,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Same turn as ``/api/chat`` but streamed, so the UI can show progress.

    Emits server-sent events: ``start`` once, ``progress`` every ~1.5s while the
    turn runs in a worker thread, then a final ``result`` (the exact payload of
    ``/api/chat``) or ``error``.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    trip_id = req.trip_id or req.session_id or "default-trip"
    thread_id = req.session_id or f"{req.user_id}:{trip_id}"

    async def run_turn() -> tuple[bool, dict[str, Any]]:
        try:
            payload = await asyncio.to_thread(
                service.run_turn,
                req.message,
                req.user_id,
                trip_id,
                req.session_id,
                req.real_only,
            )
            return True, payload
        except Exception as exc:  # noqa: BLE001 - the UI needs the failure reason
            traceback.print_exc()
            return False, {"detail": f"{type(exc).__name__}: {exc}"}

    async def events() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        started = loop.time()
        baseline = _thread_activity(thread_id)
        task = asyncio.create_task(run_turn())
        yield _sse({"type": "start", "thread_id": thread_id, "elapsed": 0.0})
        try:
            while not task.done():
                await asyncio.sleep(1.5)
                if task.done():
                    break
                snapshot = _thread_activity(thread_id)
                yield _sse(
                    {
                        "type": "progress",
                        "elapsed": round(loop.time() - started, 1),
                        "model_calls": max(0, snapshot.get("model_calls", 0) - baseline.get("model_calls", 0)),
                        "tools": max(0, snapshot.get("tools", 0) - baseline.get("tools", 0)),
                        "last_tool": snapshot.get("last_tool"),
                        "stage": snapshot.get("stage"),
                    }
                )
        except asyncio.CancelledError:  # client closed the tab
            raise
        ok, payload = await task
        elapsed = round(loop.time() - started, 1)
        yield _sse({"type": "result" if ok else "error", "elapsed": elapsed, **payload})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/api/plan")
def plan_state(session_id: str) -> dict[str, Any]:
    """Project the board plan for an existing thread (used after a page refresh)."""
    thread_id = (session_id or "").strip()
    if not thread_id:
        return {"plan": None}
    with service._lock:
        service._ensure_initialized(True)
        snapshot = service._graph.get_state({"configurable": {"thread_id": thread_id}})
        state = dict(snapshot.values or {})
        plan = _plan_view(state, service._database.latest_report(thread_id), "")
    return {"plan": plan}


@app.post("/api/cancel")
def cancel_turn(session_id: str = "") -> dict[str, Any]:
    """Ask the running turn of this session to stop at the next safe boundary."""
    return {"cancelled": service.cancel(session_id)}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    # Sessions own their trip unless the caller explicitly shares one; sharing the
    # "default-trip" row across unrelated browser sessions leaked state between users.
    trip_id = req.trip_id or req.session_id or "default-trip"
    try:
        return service.run_turn(
            message=req.message,
            user_id=req.user_id,
            trip_id=trip_id,
            session_id=req.session_id,
            real_only=req.real_only,
        )
    except Exception as exc:  # noqa: BLE001 - the UI needs the failure reason
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
