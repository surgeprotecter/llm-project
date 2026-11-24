"""One canonical itinerary: model, adapters, and the only two renderers.

Before this module the same itinerary existed in four shapes:

* ``TripState`` fields (destinations / travel_dates / hotel_options ...),
* the runtime report workspace (``report_workspace.data``),
* the saved report payload (``reports.payload_json``), and
* the chat answer (free Markdown written by the model).

The workbench therefore had to parse that answer back into structure, so the
left bubble and the right board could disagree (different stop count, different
wording, different budget lines). Everything now converts into one
:class:`Itinerary` first; the board and the chat answer are both rendered from
it, and nothing downstream reads prose.

Adapters, in priority order (see :func:`build_itinerary`):

1. ``state["itinerary"]``  — already canonical, written by the engine;
2. ``report_workspace.data`` — the planner's executed stops;
3. ``reports.payload_json`` — a rendered/legacy report row.

The approved proposal text is merged in as a last step: the approved plan is the
authority for *what the user agreed to*, so its listed segments become the day's
stops when the executed workspace is sparser (a route/schedule step may have
verified only one POI).
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

PERIODS = ("清晨", "早上", "上午", "中午", "下午", "傍晚", "晚上", "晚间", "夜里", "夜")
_TABLE_ROW = re.compile(r"^\|")
_TIME_CELL = re.compile(r"^\d{1,2}:\d{2}(?:[–~-]\d{1,2}:\d{2})?$")
_CLOCK = re.compile(r"(\d{1,2}:\d{2})(?:\s*[–~-]\s*(\d{1,2}:\d{2}))?")
_PERIOD_CELL = re.compile(r"^(" + "|".join(PERIODS) + r")$")
_STEP_LINE = re.compile(r"^(?:\d+[.、)]|[-*])\s*(.*)$")
_PERIOD_HEAD = re.compile(r"^\*\*\s*(" + "|".join(PERIODS) + r")\s*[：:]\s*(.+?)\s*\*\*\s*$")
# "上午 9:00 中环出发" / "**上午：港大本部**"
_BOLD = re.compile(r"^\*\*(.+?)\*\*\s*[：:]?\s*(.*)$")

_KIND_HINTS = (
    ("港铁", "交通"), ("地铁", "交通"), ("步行", "交通"), ("小轮", "交通"), ("巴士", "交通"),
    ("打车", "交通"), ("交通", "交通"), ("返程", "交通"), ("去程", "交通"),
    ("午餐", "美食"), ("晚餐", "美食"), ("早餐", "美食"), ("用餐", "美食"), ("早饭", "美食"),
    ("晚饭", "美食"), ("宵夜", "美食"), ("吃饭", "美食"), ("餐饮", "美食"),
    ("茶餐厅", "美食"), ("餐", "美食"), ("点心", "美食"), ("咖啡", "美食"),
    ("酒店", "住宿"), ("民宿", "住宿"), ("入住", "住宿"), ("住宿", "住宿"), ("退房", "住宿"),
)
_KIND_EN = {
    "sightseeing": "游览", "dining": "美食", "food": "美食", "restaurant": "美食",
    "transport": "交通", "transit": "交通", "hotel": "住宿", "stay": "住宿",
    "arrival": "交通", "departure": "交通", "shopping": "游览", "activity": "游览",
}
_BOARD_COLORS = ("bg-slate-800", "bg-amber-600", "bg-orange-500", "bg-teal-600",
                 "bg-stone-500", "bg-sky-600", "bg-rose-500")


def kind_of(text: str) -> str:
    """Map a Chinese/English hint to the board's four labels."""
    value = str(text or "")
    for hint, kind in _KIND_HINTS:
        if hint in value:
            return kind
    return _KIND_EN.get(value.strip().lower(), "游览")


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace("¥", "").replace("元", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


class TimeSlot(BaseModel):
    raw: str = ""
    start: str | None = None
    end: str | None = None
    period: str | None = None


class Stop(BaseModel):
    name: str
    kind: str = "游览"
    slot: TimeSlot = Field(default_factory=TimeSlot)
    duration_minutes: int | None = None
    area: str = ""
    address: str = ""
    note: str = ""
    map_url: str = ""
    refs: list[str] = Field(default_factory=list)


class Leg(BaseModel):
    mode: str = ""
    minutes: int | None = None
    note: str = ""


class Day(BaseModel):
    index: int = 1
    date: str | None = None
    city: str = ""
    theme: str = ""
    summary: str = ""
    stops: list[Stop] = Field(default_factory=list)
    legs: list[Leg] = Field(default_factory=list)


class Flight(BaseModel):
    direction: str = "去程"
    airline: str = ""
    flight_no: str = ""
    route: str = ""
    depart: str = ""
    arrive: str = ""
    price: float | None = None
    currency: str = "CNY"
    selected: bool = False


class Hotel(BaseModel):
    name: str
    area: str = ""
    rating: float | None = None
    nights: int | None = None
    price_per_night: float | None = None
    tags: list[str] = Field(default_factory=list)
    selected: bool = False
    url: str = ""


class BudgetLine(BaseModel):
    label: str
    amount: float = 0
    basis: str = ""


class Itinerary(BaseModel):
    schema_version: int = SCHEMA_VERSION
    title: str = "旅行计划"
    summary: str = ""
    destination: str = ""
    destinations: list[str] = Field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    duration_days: int = 1
    traveler_count: int = 1
    currency: str = "CNY"
    total_budget: float | None = None
    estimated_cost: float | None = None
    weather_summary: str = ""
    days: list[Day] = Field(default_factory=list)
    flights: list[Flight] = Field(default_factory=list)
    hotels: list[Hotel] = Field(default_factory=list)
    budget_lines: list[BudgetLine] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    status: str = ""

    @property
    def has_content(self) -> bool:
        return bool(self.days or self.hotels or self.flights)


def _slot_from_text(text: str) -> TimeSlot:
    raw = str(text or "").strip()
    clock = _CLOCK.search(raw)
    if clock:
        return TimeSlot(raw=raw, start=clock.group(1), end=clock.group(2))
    period = _PERIOD_CELL.match(raw.strip("*").strip())
    if period:
        return TimeSlot(raw=raw, period=period.group(1))
    if raw:
        head = re.match(r"^(" + "|".join(PERIODS) + r")\s*(.*)$", raw)
        if head:
            return TimeSlot(raw=raw, period=head.group(1))
    return TimeSlot(raw=raw)


def _stop_from_workspace(raw: dict[str, Any], fallback_area: str = "") -> Stop:
    name = str(raw.get("name") or raw.get("title") or raw.get("map_name") or "").strip()
    slot = TimeSlot(raw="", start=raw.get("start_time"), end=raw.get("end_time"))
    duration = raw.get("duration_minutes")
    return Stop(
        name=name or "未命名安排",
        kind=kind_of(str(raw.get("kind") or raw.get("type") or "")),
        slot=slot,
        duration_minutes=int(duration) if isinstance(duration, (int, float)) else None,
        area=str(raw.get("area") or fallback_area or ""),
        address=str(raw.get("address") or ""),
        note=str(raw.get("reason") or raw.get("note") or raw.get("intro") or ""),
        map_url=str(raw.get("map_url") or raw.get("source_url") or ""),
        refs=[str(raw["id"])] if raw.get("id") else [],
    )


def _stop_from_text(slot_text: str, title: str, detail: str = "") -> Stop:
    title = re.sub(r"\*\*|`", "", str(title or "")).strip()
    slot = _slot_from_text(slot_text)
    label = slot.start or slot.period or slot_text
    return Stop(name=title or label or "安排", kind=kind_of(title), slot=slot,
                note=re.sub(r"\*\*|`", "", detail or "").strip()[:160])


def stops_from_text(plan_text: str, limit: int = 14) -> list[Stop]:
    """Convert a plan proposal (table / numbered steps / period headings) into stops."""
    text = str(plan_text or "")
    table: list[Stop] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not _TABLE_ROW.match(stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2 or not (_TIME_CELL.match(cells[0]) or _PERIOD_CELL.match(cells[0])):
            continue
        table.append(_stop_from_text(cells[0].replace("–", "-"), cells[1], cells[2] if len(cells) > 2 else ""))
        if len(table) >= limit:
            return table
    if table:
        return table

    steps: list[Stop] = []
    for line in text.splitlines():
        match = _STEP_LINE.match(line.strip())
        if not match:
            continue
        body = match.group(1).strip()
        if not body:
            continue
        bold = _BOLD.match(body)
        label, rest = (bold.group(1).strip().rstrip("：:"), bold.group(2).strip()) if bold else ("", body)
        stamp = ""
        clock = _CLOCK.search(f"{label} {rest}")
        if clock:
            stamp = f"{clock.group(1)}-{clock.group(2)}" if clock.group(2) else clock.group(1)
        elif _PERIOD_CELL.match(label) or _PERIOD_CELL.match(rest):
            stamp = label if _PERIOD_CELL.match(label) else rest
        else:
            period = re.match(r"^(" + "|".join(PERIODS) + r")\s*(.*)$", rest)
            if period:
                stamp, rest = period.group(1), period.group(2)
        if not stamp:
            continue
        name = rest
        clock_head = _CLOCK.match(rest.strip())
        if clock_head:
            name = rest.strip()[clock_head.end():].strip()
        else:
            head = re.match(r"^(" + "|".join(PERIODS) + r")\s*", rest)
            if head:
                name = rest[head.end():].strip()
        steps.append(_stop_from_text(stamp, name or rest, "" if label in {"", stamp} else label))
        if len(steps) >= limit:
            return steps
    if steps:
        return steps

    sections: list[Stop] = []
    current: Stop | None = None
    for line in text.splitlines():
        stripped = line.strip()
        head = _PERIOD_HEAD.match(stripped)
        if head:
            current = _stop_from_text(head.group(1), head.group(2))
            sections.append(current)
            if len(sections) >= limit:
                break
            continue
        if current is not None and stripped.startswith("- "):
            extra = stripped[2:].strip()
            if extra:
                current.note = (current.note + " " + extra).strip()[:160]
    return [stop for stop in sections if stop.name]


def _trip_fields(trip: dict[str, Any] | None) -> dict[str, Any]:
    trip = trip or {}
    dates = trip.get("travel_dates") or {}
    budget = trip.get("budget") if isinstance(trip.get("budget"), dict) else {}
    destinations = [str(item) for item in (trip.get("destinations") or []) if item]
    amount = _number(budget.get("amount")) or _number(budget.get("max_amount"))
    return {
        "destinations": destinations,
        "destination": destinations[0] if destinations else "",
        "start_date": dates.get("start_date"),
        "end_date": dates.get("end_date"),
        "traveler_count": int(_number(trip.get("traveler_count")) or 1),
        "currency": str(budget.get("currency") or "CNY"),
        "total_budget": amount,
        "duration_days": int(_number(trip.get("duration_days")) or 1),
    }


def _hotels_from_workspace(data: dict[str, Any]) -> list[Hotel]:
    hotels: list[Hotel] = []
    for region in data.get("hotel_regions") or []:
        for item in (region or {}).get("hotels") or []:
            price = _number(item.get("nightly_price") or item.get("price") or item.get("price_per_night"))
            nights = _number(item.get("nights"))
            hotels.append(Hotel(
                name=str(item.get("name") or "住宿候选"),
                area=str(item.get("area") or item.get("location") or ""),
                rating=_number(item.get("rating") or item.get("review_score")),
                nights=int(nights) if nights else None,
                price_per_night=price,
                tags=[str(tag) for tag in (item.get("tags") or item.get("pros") or [])][:3],
                url=str(item.get("url") or item.get("source_url") or ""),
            ))
    return hotels[:4]


def _flights_from_workspace(data: dict[str, Any]) -> list[Flight]:
    flights: list[Flight] = []
    for group in data.get("transport") or []:
        for option in (group or {}).get("options") or []:
            if not (option.get("airline") or option.get("flight_no") or option.get("flight_number")):
                continue
            flights.append(Flight(
                direction=str(option.get("direction") or "去程"),
                airline=str(option.get("airline") or ""),
                flight_no=str(option.get("flight_no") or option.get("flight_number") or ""),
                route=str(option.get("route") or ""),
                depart=str(option.get("departure_time") or option.get("depart_time") or ""),
                arrive=str(option.get("arrival_time") or option.get("arrive_time") or ""),
                price=_number(option.get("price") or option.get("price_amount")),
                currency=str(option.get("currency") or "CNY"),
            ))
            if len(flights) >= 3:
                return flights
    return flights


def from_workspace(data: dict[str, Any], trip: dict[str, Any] | None = None) -> Itinerary | None:
    """Adapt ``report_workspace.data`` (the executed itinerary)."""
    data = data if isinstance(data, dict) else {}
    trip_fields = _trip_fields(trip)
    days: list[Day] = []
    for raw in data.get("days") or []:
        raw = raw or {}
        area = str(raw.get("area") or "")
        stops = [_stop_from_workspace(item, area) for item in (raw.get("stops") or []) if isinstance(item, dict)]
        if not stops:
            stops = [_stop_from_workspace(item, area) for item in (raw.get("timeline") or []) if isinstance(item, dict)]
        legs = [Leg(mode=str(leg.get("mode") or ""), minutes=int(_number(leg.get("duration_minutes")) or 0) or None,
                    note=str(leg.get("note") or ""))
                for leg in (raw.get("legs") or []) if isinstance(leg, dict)]
        days.append(Day(
            index=int(_number(raw.get("day")) or len(days) + 1),
            date=str(raw.get("date") or raw.get("start_date") or "") or None,
            city=str(raw.get("city") or raw.get("stay_city") or area),
            theme=str(raw.get("title") or ""),
            summary=str(raw.get("intro") or ""),
            stops=stops,
            legs=legs,
        ))
    budget_lines = []
    for item in ((data.get("budget") or {}).get("items") or []):
        amount = _number(item.get("amount"))
        if item.get("label") or item.get("category"):
            budget_lines.append(BudgetLine(label=str(item.get("label") or item.get("category")),
                                           amount=amount or 0, basis=str(item.get("basis") or "")))
    itinerary = Itinerary(
        **trip_fields,
        days=days,
        hotels=_hotels_from_workspace(data),
        flights=_flights_from_workspace(data),
        budget_lines=budget_lines,
        warnings=[str(item) for item in (data.get("warnings") or []) if item],
        status=str((data.get("region") or {}).get("status") or ""),
        title=(f"{trip_fields['destination']}行程" if trip_fields["destination"] else "旅行计划"),
    )
    return itinerary if itinerary.has_content else None


def from_report_payload(payload: dict[str, Any], trip: dict[str, Any] | None = None) -> Itinerary | None:
    """Adapt a saved ``reports.payload_json`` row."""
    payload = payload if isinstance(payload, dict) else {}
    report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
    trip_fields = _trip_fields({**(payload.get("trip") or {}), **(trip or {})})
    days: list[Day] = []
    for raw in report.get("days") or []:
        raw = raw or {}
        stops = [_stop_from_workspace(item, str(raw.get("area") or "")) for item in (raw.get("stops") or []) if isinstance(item, dict)]
        legs = [Leg(mode=str(leg.get("mode") or ""), minutes=int(_number(leg.get("duration_minutes")) or 0) or None,
                    note=str(leg.get("note") or ""))
                for leg in (raw.get("legs") or []) if isinstance(leg, dict)]
        days.append(Day(index=int(_number(raw.get("day")) or len(days) + 1),
                        date=str(raw.get("date") or "") or None, city=str(raw.get("city") or ""),
                        theme=str(raw.get("title") or ""), summary=str(raw.get("intro") or ""),
                        stops=stops, legs=legs))
    hotels: list[Hotel] = []
    for region in report.get("hotel_regions") or []:
        for item in (region or {}).get("hotels") or []:
            hotels.append(Hotel(name=str(item.get("name") or "住宿候选"),
                                area=str(item.get("area") or ""),
                                rating=_number(item.get("rating")),
                                nights=int(_number(item.get("nights"))) if _number(item.get("nights")) else None,
                                price_per_night=_number(item.get("nightly_price") or item.get("price")),
                                tags=[str(tag) for tag in (item.get("pros") or [])][:3],
                                url=str(item.get("url") or "")))
    flights: list[Flight] = []
    for group in report.get("transport") or []:
        for option in (group or {}).get("options") or []:
            if not (option.get("airline") or option.get("flight_no")):
                continue
            flights.append(Flight(direction=str(option.get("direction") or "去程"),
                                  airline=str(option.get("airline") or ""),
                                  flight_no=str(option.get("flight_no") or ""),
                                  route=str(option.get("route") or ""),
                                  depart=str(option.get("depart_time") or ""),
                                  arrive=str(option.get("arrive_time") or ""),
                                  price=_number(option.get("price")),
                                  currency=str(option.get("currency") or "CNY")))
    budget = report.get("budget") if isinstance(report.get("budget"), dict) else {}
    budget_lines = [BudgetLine(label=str(item.get("label") or item.get("category")),
                               amount=_number(item.get("amount")) or 0,
                               basis=str(item.get("basis") or ""))
                    for item in (budget.get("items") or []) if item.get("label") or item.get("category")]
    itinerary = Itinerary(
        **trip_fields,
        title=str(payload.get("title") or report.get("title")
                  or (f"{trip_fields['destination']}行程" if trip_fields["destination"] else "旅行计划")),
        summary=str(report.get("summary") or payload.get("summary") or ""),
        days=days, hotels=hotels, flights=flights,
        estimated_cost=_number(budget.get("total")) or _number(payload.get("total_estimated_cost")),
        budget_lines=budget_lines,
        warnings=[str(item) for item in (report.get("warnings") or []) if item],
        sources=[item for item in (report.get("sources") or []) if isinstance(item, dict)],
        status=str(report.get("status") or ""),
    )
    return itinerary if itinerary.has_content else None


def merge_proposal(itinerary: Itinerary, proposal: str) -> Itinerary:
    """Let the approved proposal fill a sparse executed day (same trip, one day)."""
    stops = stops_from_text(proposal)
    if len(stops) < 2:
        return itinerary
    if itinerary.duration_days > 1:
        return itinerary
    if itinerary.days and sum(len(day.stops) for day in itinerary.days) >= len(stops):
        return itinerary
    day = itinerary.days[0] if itinerary.days else Day(index=1)
    day = day.model_copy(update={"stops": stops})
    return itinerary.model_copy(update={"days": [day]})


def build_itinerary(result: dict[str, Any] | None, report_payload: dict[str, Any] | None = None) -> Itinerary | None:
    """Resolve every representation of the same trip into one canonical object."""
    result = result if isinstance(result, dict) else {}
    stored = result.get("itinerary")
    itinerary: Itinerary | None = None
    if isinstance(stored, dict) and stored.get("days") is not None:
        try:
            itinerary = Itinerary.model_validate(stored)
        except ValueError:
            itinerary = None
    if itinerary is None:
        workspace = result.get("report_workspace") if isinstance(result.get("report_workspace"), dict) else {}
        itinerary = from_workspace(workspace.get("data") or {}, result.get("trip_state"))
    if itinerary is None and report_payload:
        itinerary = from_report_payload(report_payload, result.get("trip_state"))
    if itinerary is None:
        return None
    approval = result.get("plan_approval") if isinstance(result.get("plan_approval"), dict) else {}
    pending = result.get("pending_plan") if isinstance(result.get("pending_plan"), dict) else {}
    proposal = str(approval.get("proposal") or pending.get("proposal") or "")
    if proposal:
        itinerary = merge_proposal(itinerary, proposal)
    return itinerary


def render_board(itinerary: Itinerary) -> dict[str, Any]:
    """Project the canonical itinerary into the workbench ``plan`` schema."""
    days = []
    for day in itinerary.days:
        activities = [{
            "time": stop.slot.start or stop.slot.end or stop.slot.period or stop.slot.raw or "待定",
            "type": stop.kind,
            "title": stop.name,
            "location": stop.area or stop.address,
            "duration": f"{stop.duration_minutes} 分钟" if stop.duration_minutes else "",
            "desc": stop.note,
            "tip": "",
        } for stop in day.stops]
        days.append({
            "day": day.index,
            "date": day.date or "",
            "theme": day.theme or (f"第 {day.index} 天"),
            "summary": day.summary,
            "activities": activities,
        })
    breakdown = [{"label": line.label, "amount": line.amount, "color": _BOARD_COLORS[index % len(_BOARD_COLORS)]}
                 for index, line in enumerate(itinerary.budget_lines)]
    estimated = itinerary.estimated_cost
    if estimated is None and breakdown:
        estimated = float(sum(line["amount"] for line in breakdown))
    return {
        "id": f"{itinerary.title}-live",
        "title": itinerary.title,
        "subtitle": itinerary.summary or "由 LangGraph 实时 runtime 生成；未核验的价格与库存以下单页面为准。",
        "destination": itinerary.destination or "待确认",
        "duration": f"{itinerary.duration_days} 天",
        "travelers": f"{itinerary.traveler_count} 位旅行者",
        "totalBudget": itinerary.total_budget or estimated or 0,
        "estimatedCost": estimated or itinerary.total_budget or 0,
        "currency": itinerary.currency,
        "weatherSummary": itinerary.weather_summary,
        "budgetBreakdown": breakdown,
        "flights": [{
            "direction": flight.direction, "airline": flight.airline,
            "route": flight.route or flight.flight_no,
            "duration": "", "price": f"¥{flight.price:,.0f}" if flight.price else "",
            "selected": flight.selected,
        } for flight in itinerary.flights],
        "hotels": [{
            "name": hotel.name, "area": hotel.area, "rating": hotel.rating or 0,
            "nights": hotel.nights or itinerary.duration_days - 1,
            "pricePerNight": int(hotel.price_per_night) if hotel.price_per_night else 0,
            "tags": hotel.tags, "selected": hotel.selected,
        } for hotel in itinerary.hotels],
        "days": days,
    }


def render_chat(itinerary: Itinerary) -> str:
    """Render the same itinerary as the chat answer (Markdown), never model prose."""
    header = f"**{itinerary.title}**"
    meta = [value for value in (
        itinerary.destination,
        f"{itinerary.duration_days} 天",
        f"{itinerary.traveler_count} 人",
        f"预算上限 ¥{itinerary.total_budget:,.0f}" if itinerary.total_budget else "",
    ) if value]
    lines = [header, "", " · ".join(meta), ""]
    if itinerary.days:
        lines.extend(["| 时间 | 安排 | 说明 |", "|---|---|---|"])
        for day in itinerary.days:
            for stop in day.stops:
                if day.index > 1:
                    continue
                stamp = stop.slot.start or stop.slot.period or "待定"
                if stop.slot.end:
                    stamp = f"{stamp}–{stop.slot.end}"
                note = stop.note.replace("|", "/")[:120]
                lines.append(f"| {stamp} | {stop.name} | {note} |")
        lines.append("")
    if itinerary.budget_lines:
        lines.extend(["**预算**", ""])
        for line in itinerary.budget_lines:
            lines.append(f"- {line.label}：¥{line.amount:,.0f}")
        lines.append("")
    if itinerary.hotels:
        lines.extend(["**住宿候选**", ""])
        for hotel in itinerary.hotels:
            price = f"约 ¥{hotel.price_per_night:,.0f}/晚" if hotel.price_per_night else "价格待核"
            lines.append(f"- {hotel.name}（{hotel.area or '位置待核'}）：{price}")
        lines.append("")
    if itinerary.flights:
        lines.extend(["**航班候选**", ""])
        for flight in itinerary.flights:
            price = f"¥{flight.price:,.0f}" if flight.price else "价格待核"
            lines.append(f"- {flight.direction} {flight.airline} {flight.flight_no}：{flight.route}，{price}")
        lines.append("")
    if itinerary.warnings:
        lines.extend(["**注意事项**", "", *[f"- {item}" for item in itinerary.warnings], ""])
    return "\n".join(lines).strip()


def preview_from_proposal(trip: dict[str, Any] | None, proposal: str) -> Itinerary | None:
    """Board preview of an approved/pending proposal before execution finishes.

    Execution can stall (or the user may confirm and walk away); the workbench
    still has to show the plan the user agreed to. This preview is explicitly
    marked as unexecuted and is replaced by the canonical itinerary as soon as
    the schedule step writes real stops.
    """
    stops = stops_from_text(proposal)
    if len(stops) < 2:
        return None
    fields = _trip_fields(trip)
    destination = fields["destination"] or "待确认"
    return Itinerary(
        **fields,
        title=f"{destination}行程（待执行）",
        summary="已批准的行程草案；执行完成后会替换为已核实的时间轴。",
        days=[Day(index=1, date=fields["start_date"], city=fields["destination"],
                  theme="已批准方案", summary="按提案整理，等待执行核实。", stops=stops)],
    )
