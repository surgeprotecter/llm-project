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
import re
import sqlite3
import threading
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import date, timedelta
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

_BUDGET_COLORS = [
    "bg-slate-800",
    "bg-amber-600",
    "bg-orange-500",
    "bg-teal-600",
    "bg-stone-500",
    "bg-sky-600",
    "bg-rose-500",
]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _budget_breakdown(
    report: dict[str, Any], payload: dict[str, Any], trip: dict[str, Any]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push(label: Any, amount: Any) -> None:
        text = str(label or "").strip()
        value = _num(amount, -1.0)
        if not text or value <= 0 or text in seen:
            return
        seen.add(text)
        items.append(
            {"label": text, "amount": round(value), "color": _BUDGET_COLORS[len(items) % len(_BUDGET_COLORS)]}
        )

    budget = report.get("budget") if isinstance(report.get("budget"), dict) else {}
    for item in budget.get("items") or []:
        if isinstance(item, dict):
            push(item.get("label") or item.get("category"), item.get("amount"))
    raw_breakdown = payload.get("budget_breakdown") or trip.get("budget_breakdown") or {}
    if isinstance(raw_breakdown, dict):
        for label, amount in raw_breakdown.items():
            push(label, amount)
    return items


def _weather_summary(payload: dict[str, Any], report: dict[str, Any]) -> str:
    weather = payload.get("weather")
    if isinstance(weather, dict) and weather:
        parts = [f"{key}：{value}" for key, value in list(weather.items())[:4] if value not in (None, "")]
        if parts:
            return "；".join(str(part) for part in parts)
    days = report.get("days") or []
    for day in days:
        if isinstance(day, dict) and day.get("weather"):
            return str(day["weather"])
    return "实时运行结果未返回天气；出行前请重新查询。"


def _activity_view(raw: Any, fallback_area: str) -> dict[str, str]:
    if isinstance(raw, str):
        raw = {"name": raw}
    if not isinstance(raw, dict):
        raw = {}
    hours = raw.get("estimated_hours")
    duration = str(raw.get("duration") or "")
    if not duration and hours:
        duration = f"{hours} 小时"
    minutes = raw.get("duration_minutes")
    if not duration and minutes:
        try:
            duration = f"{int(minutes)} 分钟"
        except (TypeError, ValueError):
            duration = ""
    return {
        "time": str(raw.get("start_time") or raw.get("time") or ""),
        "type": _TYPE_LABELS.get(str(raw.get("type") or raw.get("category") or "").lower(), str(raw.get("type") or raw.get("category") or "活动")),
        "title": str(raw.get("name") or raw.get("title") or "未命名活动"),
        "location": str(raw.get("area") or raw.get("location") or raw.get("address") or fallback_area or ""),
        "duration": duration,
        "desc": str(raw.get("reason") or raw.get("desc") or raw.get("content") or ""),
        "tip": str(raw.get("source_note") or raw.get("tip") or ""),
    }


def _day_views(days: Any, start_date: Any) -> list[dict[str, Any]]:
    if not isinstance(days, list):
        return []
    origin = None
    if isinstance(start_date, str) and start_date:
        try:
            origin = date.fromisoformat(start_date)
        except ValueError:
            origin = None
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(days):
        if not isinstance(raw, dict):
            continue
        day_no = int(_num(raw.get("day"), index + 1))
        date_text = str(raw.get("date") or "")
        if not date_text and origin is not None:
            date_text = (origin + timedelta(days=max(day_no - 1, 0))).strftime("%m月%d日")
        area = str(raw.get("area") or raw.get("city") or "")
        stops = raw.get("stops") or raw.get("activities") or []
        activities = [_activity_view(stop, area) for stop in stops[:10]] if isinstance(stops, list) else []
        result.append(
            {
                "day": day_no,
                "date": date_text or f"第 {day_no} 天",
                "theme": str(raw.get("title") or f"第 {day_no} 天"),
                "summary": str(raw.get("intro") or raw.get("summary") or ""),
                "activities": activities,
            }
        )
    return result


def _hotel_views(report: dict[str, Any], payload: dict[str, Any], trip: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[Any] = []
    for group in report.get("hotel_regions") or []:
        if isinstance(group, dict):
            records.extend(group.get("hotels") or [])
    records.extend(payload.get("hotel_options") or [])
    records.extend(trip.get("hotel_options") or [])
    nights = int(_num(trip.get("duration_days"), 1) or 1)
    views: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or raw.get("title") or raw.get("hotel_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        price = _num(raw.get("nightly_price") or raw.get("price_per_night") or raw.get("price"))
        total = _num(raw.get("total_price") or raw.get("totalPrice"))
        if not price and total and nights:
            price = total / nights
        tags = [str(tag) for tag in (raw.get("tags") or raw.get("pros") or []) if str(tag).strip()][:4]
        views.append(
            {
                "name": name,
                "area": str(raw.get("area") or raw.get("location") or raw.get("address") or ""),
                "rating": str(raw.get("rating") or raw.get("score") or "—"),
                "nights": nights,
                "pricePerNight": round(price),
                "tags": tags or ["需核对房型与可订状态"],
                "selected": bool(raw.get("selected", False)),
            }
        )
        if len(views) >= 4:
            break
    if views:
        views[0]["selected"] = True
    return views


def _flight_views(report: dict[str, Any], payload: dict[str, Any], trip: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[Any] = []
    for group in report.get("transport") or []:
        if isinstance(group, dict):
            records.extend(group.get("options") or [])
    records.extend(payload.get("transport_options") or [])
    records.extend(trip.get("transport_options") or [])
    views: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            continue
        if raw.get("train_no") or str(raw.get("mode") or "").lower() in {"rail", "train"}:
            continue
        airline = str(raw.get("airline") or raw.get("carrier") or raw.get("flight_no") or "").strip()
        if not airline:
            continue
        route = str(
            raw.get("route")
            or " → ".join(
                str(part)
                for part in (
                    raw.get("origin") or raw.get("from"),
                    raw.get("destination") or raw.get("to"),
                )
                if part
            )
            or ""
        )
        price = raw.get("price") or raw.get("total_price") or raw.get("amount")
        price_text = f"¥{round(_num(price)):,}" if _num(price) > 0 else "价格待查询"
        key = f"{airline}|{route}"
        if key in seen:
            continue
        seen.add(key)
        views.append(
            {
                "direction": str(raw.get("direction") or raw.get("date") or "交通候选"),
                "airline": airline,
                "route": route or "航线待确认",
                "duration": str(raw.get("duration") or raw.get("stops") or ""),
                "price": price_text,
                "selected": bool(raw.get("selected", False)),
            }
        )
        if len(views) >= 3:
            break
    if views:
        views[0]["selected"] = True
    return views


_TIME_CELL = re.compile(r"^\d{1,2}:\d{2}(?:[–~-]\d{1,2}:\d{2})?$")
_PERIOD_CELL = re.compile(r"^(清晨|早上|上午|中午|下午|傍晚|晚上|晚间|夜里|夜)$")
_STEP_LINE = re.compile(r"^(?:\d+[.、)]|[-*])\s*(.*)$")
_CLOCK = re.compile(r"(\d{1,2}:\d{2}(?:[–~-]\d{1,2}:\d{2})?)")
_ACTIVITY_HINTS = (
    ("港铁", "交通"), ("步行", "交通"), ("小轮", "交通"), ("交通", "交通"),
    ("午餐", "美食"), ("晚餐", "美食"), ("早餐", "美食"), ("冰厅", "美食"),
    ("用餐", "美食"), ("早饭", "美食"), ("晚饭", "美食"), ("宵夜", "美食"),
    ("吃饭", "美食"), ("餐饮", "美食"),
    ("茶餐厅", "美食"), ("点心", "美食"), ("夜市", "游览"),
)
_TYPE_LABELS = {"sightseeing": "游览", "dining": "美食", "food": "美食",
                "transport": "交通", "hotel": "住宿", "stay": "住宿"}


def _activity_kind(text: str) -> str:
    for hint, kind in _ACTIVITY_HINTS:
        if hint in text:
            return kind
    return "游览"


_PERIOD_HEAD = re.compile(
    r"^\*\*\s*(清晨|早上|上午|中午|下午|傍晚|晚上|晚间|夜里|夜)\s*[：:]\s*(.+?)\s*\*\*\s*$"
)


def _period_sections(plan_text: str) -> list[dict[str, str]]:
    """Parse proposals written as ``**上午：港大本部**`` headings plus bullet lines."""
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in (plan_text or "").splitlines():
        line = raw.strip()
        head = _PERIOD_HEAD.match(line)
        if head:
            title = head.group(2).strip()
            current = {
                "time": head.group(1),
                "type": _activity_kind(title),
                "title": title,
                "location": "",
                "duration": "",
                "desc": "",
                "tip": "",
            }
            rows.append(current)
            if len(rows) >= 14:
                break
            continue
        if current is not None and line.startswith("- "):
            extra = line[2:].strip()
            if extra:
                current["desc"] = (current["desc"] + " " + extra).strip()[:120]
    return [row for row in rows if row["title"]]


def _reply_table_activities(reply: str) -> list[dict[str, str]]:
    """Extract timetable rows (``| 09:00–09:45 | 安排 | 交通 |``) from a chat answer."""
    rows: list[dict[str, str]] = []
    for line in (reply or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2 or not (_TIME_CELL.match(cells[0]) or _PERIOD_CELL.match(cells[0])):
            continue
        title = re.sub(r"\*\*|`", "", cells[1]).strip()
        if not title:
            continue
        detail = re.sub(r"\*\*|`", "", cells[2]).strip() if len(cells) > 2 else ""
        rows.append({
            "time": cells[0].replace("–", "-"),
            "type": _activity_kind(title),
            "title": title,
            "location": "",
            "duration": "",
            "desc": detail,
            "tip": "",
        })
        if len(rows) >= 14:
            break
    return rows


def _step_activities(plan_text: str) -> list[dict[str, str]]:
    """Parse numbered/bulleted plan steps that carry a clock time or a period word."""
    rows: list[dict[str, str]] = []
    for line in (plan_text or "").splitlines():
        match = _STEP_LINE.match(line.strip())
        if not match:
            continue
        body = match.group(1).strip()
        if not body:
            continue
        label = ""
        rest = body
        bold = re.match(r"^\*\*(.+?)\*\*\s*[:：]?\s*(.*)$", body)
        if bold:
            label = bold.group(1).strip().rstrip("：:")
            rest = bold.group(2).strip()
        clock = _CLOCK.search(f"{label} {rest}")
        if clock:
            stamp = clock.group(1).replace("–", "-")
        else:
            period = _PERIOD_CELL.match(label) or _PERIOD_CELL.match(rest)
            if not period:
                continue
            stamp = period.group(1)
        title = re.sub(r"\*\*|`", "", rest or label).strip()
        if not title:
            continue
        rows.append({
            "time": stamp,
            "type": _activity_kind(title),
            "title": title,
            "location": "",
            "duration": "",
            "desc": label if label and label != stamp else "",
            "tip": "",
        })
        if len(rows) >= 14:
            break
    return rows


def _plan_view(
    result: dict[str, Any], report_row: dict[str, Any] | None, reply: str = ""
) -> dict[str, Any] | None:
    """Project a live runtime turn into the workbench ``plan`` schema."""
    trip = result.get("trip_state") if isinstance(result.get("trip_state"), dict) else {}
    payload = _report_payload(report_row)
    report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
    # Chat-delivery turns never write a report row: the itinerary lives in the
    # runtime workspace instead, so fold it in before projecting the board.
    workspace = result.get("report_workspace") if isinstance(result.get("report_workspace"), dict) else {}
    data = workspace.get("data") if isinstance(workspace.get("data"), dict) else {}
    if data:
        report = dict(report)
        if not report.get("days"):
            report["days"] = data.get("days") or []
        if not report.get("hotel_regions"):
            report["hotel_regions"] = data.get("hotel_regions") or []
        if not report.get("transport"):
            report["transport"] = data.get("transport") or []
        if not report.get("budget") and isinstance(data.get("budget"), dict):
            report["budget"] = data["budget"]

    destinations = trip.get("destinations") or []
    destination = str(payload.get("destination") or (destinations[0] if destinations else "") or "").strip()
    days_src = report.get("days") or payload.get("itinerary") or []
    hotels = _hotel_views(report, payload, trip)
    flights = _flight_views(report, payload, trip)
    if not destination and not days_src and not hotels and not flights:
        return None

    budget = trip.get("budget") if isinstance(trip.get("budget"), dict) else {}
    start_date = (trip.get("travel_dates") or {}).get("start_date")
    days = _day_views(days_src, start_date)
    approval = result.get("plan_approval") if isinstance(result.get("plan_approval"), dict) else {}
    pending = result.get("pending_plan") if isinstance(result.get("pending_plan"), dict) else {}
    proposal = str(approval.get("proposal") or pending.get("proposal") or "")
    if (
        (reply or proposal)
        and int(_num(trip.get("duration_days"), 1) or 1) <= 1
        and sum(len(day.get("activities") or []) for day in days) <= 1
    ):
        # Chat-delivery turns answer in prose: reuse the answer's timetable so the
        # board reflects what the user just read instead of one researched stop.
        parsed = _reply_table_activities(reply)
        if len(parsed) < 3:
            parsed = _step_activities(proposal) or _period_sections(proposal) or parsed
        if len(parsed) >= 3:
            days = [{
                "day": 1,
                "date": str(start_date or ""),
                "theme": f"{destination or '目的地'}当日安排",
                "summary": "按本轮答复整理的时间轴摘要。",
                "activities": parsed,
            }]
    breakdown = _budget_breakdown(report, payload, trip)
    estimated = _num(
        (report.get("budget") or {}).get("total") if isinstance(report.get("budget"), dict) else None
    ) or _num(payload.get("total_estimated_cost"))
    if not estimated and breakdown:
        estimated = float(sum(item["amount"] for item in breakdown))
    total_budget = _num(budget.get("amount") or budget.get("max_amount")) or estimated
    traveler_count = int(_num(trip.get("traveler_count"), 1) or 1)
    duration_days = int(_num(trip.get("duration_days"), len(days) or 1) or 1)

    return {
        "id": f"{trip.get('trip_id') or 'trip'}-live",
        "title": str(payload.get("title") or (f"{destination}行程" if destination else "旅行计划")),
        "subtitle": str(
            report.get("summary")
            or payload.get("summary")
            or "由 LangGraph 实时 runtime 生成；未核验的价格与库存以下单页面为准。"
        ),
        "destination": destination or "待确认",
        "duration": f"{duration_days} 天",
        "travelers": f"{traveler_count} 位旅行者",
        "totalBudget": total_budget,
        "estimatedCost": estimated,
        "currency": str(budget.get("currency") or "CNY"),
        "weatherSummary": _weather_summary(payload, report),
        "budgetBreakdown": breakdown,
        "flights": flights,
        "hotels": hotels,
        "days": days,
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
