"""Structured episode memory: one compact record per finished trip.

An episode is intentionally small and factual.  It is written when a turn
reaches a terminal state, so a later session can answer "continue last time"
without replaying the whole transcript.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from travel_agent.models import MemoryItem


def _text(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return text[:limit]


def _dates(trip: dict[str, Any]) -> tuple[str, str]:
    travel_dates = trip.get("travel_dates") or {}
    start = str(travel_dates.get("start_date") or "")
    end = str(travel_dates.get("end_date") or start)
    return start, end


def build_episode(state: dict[str, Any], *, status: str | None = None) -> MemoryItem | None:
    """Build an episode memory for a terminal turn, or None when there is nothing to keep."""
    trip = state.get("trip_state") if isinstance(state.get("trip_state"), dict) else {}
    destinations = [str(item) for item in (trip.get("destinations") or []) if str(item).strip()]
    summary = state.get("semantic_summary") if isinstance(state.get("semantic_summary"), dict) else {}
    decisions = [_text(item, 120) for item in (summary.get("decisions") or [])][:5]
    findings = [_text(item, 140) for item in (summary.get("findings") or [])][:5]
    plan = state.get("task_plan") if isinstance(state.get("task_plan"), list) else []
    open_items = [
        _text(item.get("goal") or item.get("title") or item, 120)
        for item in plan
        if isinstance(item, dict) and item.get("status") != "completed"
    ][:4]
    approval = state.get("plan_approval") if isinstance(state.get("plan_approval"), dict) else {}
    scope = state.get("request_scope") if isinstance(state.get("request_scope"), dict) else {}
    goal = _text(state.get("active_goal"), 160)
    if not goal:
        proposal = _text(approval.get("proposal") or (state.get("pending_plan") or {}).get("proposal"), 240)
        goal = proposal
    goals = [item for item in [goal, _text(scope.get("artifact"), 40) and f"交付:{scope.get('artifact')}"] if item]
    if not destinations and not decisions and not findings:
        return None

    start, end = _dates(trip)
    title = " / ".join(destinations) or "未命名行程"
    days = trip.get("duration_days")
    travelers = trip.get("traveler_count")
    budget = trip.get("budget") if isinstance(trip.get("budget"), dict) else {}
    payload = {
        "title": title,
        "destinations": destinations,
        "start_date": start,
        "end_date": end,
        "duration_days": days,
        "travelers": travelers,
        "budget": budget.get("amount") if isinstance(budget, dict) else None,
        "currency": budget.get("currency") if isinstance(budget, dict) else None,
        "goals": goals,
        "decisions": decisions,
        "verified_facts": findings,
        "open_items": open_items,
        "status": status or state.get("run_status") or "unknown",
        "plan_status": approval.get("status"),
    }
    line = f"{title}"
    if days:
        line += f" {days}天"
    if travelers:
        line += f" {travelers}人"
    if start:
        line += f" {start}"
        if end and end != start:
            line += f"～{end}"
    if decisions:
        line += "；" + "；".join(decisions[:2])

    valid_until = None
    if end:
        try:
            valid_until = datetime.combine(
                date.fromisoformat(end) + timedelta(days=30), datetime.min.time(), UTC
            )
        except ValueError:
            valid_until = None
    trip_id = state.get("trip_id") or trip.get("trip_id")
    return MemoryItem(
        user_id=str(state.get("user_id") or "default"),
        trip_id=str(trip_id) if trip_id else None,
        kind="episode",
        content=_text(line, 300),
        key=f"episode:{trip_id}" if trip_id else "",
        structured_value=payload,
        source="system",
        importance=0.8,
        confidence=0.9,
        tags=["episode"],
        valid_until=valid_until,
    )


def episode_line(item: MemoryItem) -> str:
    """One compact context line for the ``recent_trips`` partition."""
    payload = item.structured_value or {}
    parts = [str(payload.get("title") or item.content)]
    goals = payload.get("goals") or []
    decisions = payload.get("decisions") or []
    open_items = payload.get("open_items") or []
    if goals:
        parts.append(f"目标：{_text(goals[0], 80)}")
    if decisions:
        parts.append("决策：" + "；".join(_text(item, 60) for item in decisions[:2]))
    if open_items:
        parts.append("未完成：" + "；".join(_text(item, 50) for item in open_items[:2]))
    return " ｜ ".join(parts)[:400]
