from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from travel_agent.context import ContextBuilder
from travel_agent.llm import ModelGateway
from travel_agent.models import EvidenceItem, ItineraryDay, MemoryItem, TravelPlan, TripState
from travel_agent.persistence import Database
from travel_agent.reports import ReportRenderer
from travel_agent.tools import ToolRegistry


def _last_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _parse_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    mapping = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if value in mapping:
        return mapping[value]
    if len(value) == 2 and value[0] == "十" and value[1] in mapping:
        return 10 + mapping[value[1]]
    if len(value) == 2 and value[1] == "十" and value[0] in mapping:
        return mapping[value[0]] * 10
    return None


def _extract_int(text: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            parsed = _parse_number(match.group(1))
            if parsed is not None:
                return parsed
    return None


def _extract_amount(text: str) -> float | None:
    range_match = re.search(r"(?:预算|花费|费用)[^0-9]{0,8}(\d+(?:\.\d+)?)\s*(?:-|~|到|至)\s*(\d+(?:\.\d+)?)\s*(万|w|元|块)?", text, re.I)
    if not range_match:
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|~|到|至)\s*(\d+(?:\.\d+)?)\s*(万|w|元|块)", text, re.I)
    if range_match:
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        if "万" in (range_match.group(3) or "") or "w" in (range_match.group(3) or "").lower():
            lower *= 10000
            upper *= 10000
        return (lower + upper) / 2
    match = re.search(r"(?:预算|花费|费用|总共|总预算)[^0-9]{0,8}(\d+(?:\.\d+)?)\s*(?:万|w|元|块)?", text, re.I)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:万|w|元|块)", text, re.I)
    if not match:
        return None
    amount = float(match.group(1))
    if "万" in match.group(0) or "w" in match.group(0).lower():
        amount *= 10000
    return amount


def _extract_budget_range(text: str) -> tuple[float, float] | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|~|到|至)\s*(\d+(?:\.\d+)?)\s*(万|w|元|块)", text, re.I)
    if not match:
        return None
    lower, upper = float(match.group(1)), float(match.group(2))
    if "万" in match.group(3) or "w" in match.group(3).lower():
        lower *= 10000
        upper *= 10000
    return min(lower, upper), max(lower, upper)


def _flight_only_request(text: str) -> bool:
    """A flight refinement can mention the rest of the trip as unchanged."""
    clauses = re.split(r"[，,。；;！!？?\n]+|但是|不过", text)
    active = [c for c in clauses if not re.search(r"不变|保留|照旧|不用|无需|不需要|不查|不要查|已订", c)]
    focus = "，".join(active)
    return bool(re.search(r"机票|航班|飞机|商务舱|公务舱|头等舱|经济舱|flight|business class", focus, re.I)) and not bool(
        re.search(r"酒店|住宿|民宿|餐饮|景点|攻略|整[个体份]|总预算|全程预算|完整|重新规划", focus)
    )


def _trip_budget_text(text: str) -> str:
    # An airfare/room/meal allowance must never replace the trip allowance.
    clauses = re.split(r"[，,。；;！!？?\n]+", text)
    if _flight_only_request(text):
        return "，".join(c for c in clauses if re.search(r"总预算|全程预算|旅行预算|旅游预算|整体预算", c))
    return "，".join(
        clause for clause in clauses
        if not (
            re.search(r"机票|航班|飞机|酒店|住宿|民宿|餐费|餐饮|门票|flight|hotel", clause, re.I)
            and not re.search(r"总预算|全程预算|旅行预算|旅游预算|整体预算|总共|一共", clause)
        )
    )


def _budget_values(text: str) -> dict[str, float | None] | None:
    # Accept bare numbers only when explicitly attached to a budget/basis/bound.
    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(万|w)?\s*[-~～—–到至]\s*(\d+(?:\.\d+)?)\s*(万|w|元|块)?", text, re.I
    )
    if range_match and (range_match.group(4) or re.search(r"预算|人均|每人|费用|花费", text)):
        low = float(range_match.group(1)) * (10000 if (range_match.group(2) or range_match.group(4) or "").lower() in {"万", "w"} else 1)
        high = float(range_match.group(3)) * (10000 if (range_match.group(4) or "").lower() in {"万", "w"} else 1)
        low, high = sorted((low, high))
        return {"amount": (low + high) / 2, "min_amount": low, "max_amount": high}
    match = re.search(
        r"(?:预算|花费|费用|总共|人均|每人|至少|不低于|不超过|最多)[^\d，,。；;\n]{0,12}(\d+(?:\.\d+)?)\s*(万|w|元|块)?", text, re.I
    ) or re.search(r"(\d+(?:\.\d+)?)\s*(万|w|元|块)", text, re.I)
    if not match:
        return None
    value = float(match.group(1)) * (10000 if (match.group(2) or "").lower() in {"万", "w"} else 1)
    if re.search(r"至少|起码|不低于|最低|起步|以上|起[，,。\s]|上不封顶", text):
        return {"amount": None, "min_amount": value, "max_amount": None}
    if re.search(r"最多|不超过|至多|上限|以内|以下|封顶", text):
        return {"amount": None, "min_amount": None, "max_amount": value}
    return {"amount": value, "min_amount": None, "max_amount": None}


def _hong_kong_today() -> date:
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).date()


def _extract_travel_dates(text: str, reference_date: date | None = None) -> tuple[date | None, date | None]:
    today = reference_date or _hong_kong_today()
    explicit_year = re.search(r"(20\d{2})\s*年", text)
    year = int(explicit_year.group(1)) if explicit_year else today.year
    date_pattern = re.compile(r"(?:(20\d{2})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")
    found = list(date_pattern.finditer(text))
    if len(found) >= 2:
        first, second = found[0], found[1]
        start_year = int(first.group(1) or year)
        end_year = int(second.group(1) or start_year)
        try:
            start = date(start_year, int(first.group(2)), int(first.group(3)))
            end = date(end_year, int(second.group(2)), int(second.group(3)))
        except ValueError:
            return None, None
        if not first.group(1) and not explicit_year and start < today:
            start = start.replace(year=start.year + 1)
            if not second.group(1):
                end = end.replace(year=start.year)
        if end < start and not second.group(1):
            end = date(end_year + 1, end.month, end.day)
        return start, end
    iso_dates = re.findall(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", text)
    if len(iso_dates) >= 2:
        try:
            return date(*map(int, iso_dates[0])), date(*map(int, iso_dates[1]))
        except ValueError:
            return None, None
    if len(iso_dates) == 1:
        try:
            return date(*map(int, iso_dates[0])), None
        except ValueError:
            return None, None
    if found:
        item = found[0]
        try:
            start = date(int(item.group(1) or year), int(item.group(2)), int(item.group(3)))
        except ValueError:
            return None, None
        if not item.group(1) and not explicit_year and start < today:
            start = start.replace(year=start.year + 1)
        compact_end = re.match(r"\s*(?:到|至|[-—–~～])\s*(\d{1,2})\s*[日号]?(?!\s*月)", text[item.end():])
        if compact_end:
            try:
                end = date(start.year, start.month, int(compact_end.group(1)))
            except ValueError:
                return start, None
            if end < start:
                month = start.month % 12 + 1
                try:
                    end = date(start.year + (start.month == 12), month, end.day)
                except ValueError:
                    return start, None
            return start, end
        return start, None
    short_dates = list(re.finditer(r"(?<![\d/.-])(\d{1,2})[/-](\d{1,2})(?![\d/.-])", text))
    if short_dates:
        try:
            start = date(today.year, int(short_dates[0].group(1)), int(short_dates[0].group(2)))
            if start < today:
                start = start.replace(year=start.year + 1)
            end = date(start.year, int(short_dates[1].group(1)), int(short_dates[1].group(2))) if len(short_dates) > 1 else None
            if end and end < start:
                end = end.replace(year=end.year + 1)
            return start, end
        except ValueError:
            return None, None
    return None, None


def _extract_destination(text: str) -> str | None:
    text = re.sub(
        r"在\s*(?:小红书|网上|网络|网站|百度|Google)\s*(?:搜索|搜一下|搜一搜|搜|查找)",
        "搜索",
        text,
        flags=re.IGNORECASE,
    )
    city_days = list(dict.fromkeys(re.findall(r"(?:^|[，,、；;\s]|去|到|前往)([\u4e00-\u9fff]{2,8}?)\s*[0-9一二两三四五六七八九十]+\s*天", text)))
    if len(city_days)>1:
        return "与".join(city_days)
    patterns = [
        r"(?:去|到|前往|目的地[：:是为]?|搜索|搜一下|搜一搜)\s*([\u4e00-\u9fff]{2,8}?)(?=玩|旅游|旅行|度假|攻略|景点|美食|避坑|[0-9一二两三四五六七八九十]+[天日]|，|,|。|\n|$)",
        r"在\s*([\u4e00-\u9fff]{2,8}?)(?=玩|旅游|旅行|度假|攻略|景点|美食|避坑|，|,|。|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1)
            if any(word in candidate for word in ("住宿","餐饮","酒店","民宿","交通","航班","机票","餐厅","小吃","咖啡","报告")):
                continue
            if any(phrase in candidate for phrase in ["一个", "地方", "哪里", "什么", "国内旅游", "走走", "散心", "看看", "出去", "舒服"]):
                continue
            if candidate not in {"一个地方", "哪里", "什么地方", "国内旅游"}:
                return candidate
    return None


def _extract_origin(text: str) -> str | None:
    match = re.search(r"(?:从|出发地[：:是为]?)\s*([\u4e00-\u9fff]{2,8}?)(?=出发|前往|去|，|,|。|\n|$)", text)
    if not match:
        return None
    return match.group(1).removesuffix("出发")


def _extract_preferences(text: str) -> tuple[list[str], list[str]]:
    preferences: list[str] = []
    constraints: list[str] = []
    for phrase in ["不太冷", "不太热", "不要太冷", "不要太热", "小众", "不要大众景点", "美食", "自然风景", "轻松", "慢旅行", "亲子", "适合老人", "舒服", "舒适", "商务舱", "公务舱", "头等舱", "超级经济舱", "经济舱", "直飞", "腿部空间"]:
        if phrase in text:
            preferences.append(phrase)
    for phrase in ["不能去冷的地方", "不能太累", "不想爬山", "不想早起", "预算不能超", "不要红眼航班", "不坐廉航", "不要廉航", "不想转机", "不坐经济舱"]:
        if phrase in text:
            constraints.append(phrase)
    return preferences, constraints


def _merge_requirements(state: TripState, query: str) -> TripState:
    destination = _extract_destination(query)
    origin = _extract_origin(query)
    duration_text = re.sub(
        r"(?:(?:20\d{2})\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*[日号]?(?:\s*[至到—–-]\s*\d{1,2}\s*[日号]?)?",
        " ",
        query,
    )
    duration_text = re.sub(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b", " ", duration_text)
    duration = _extract_int(
        duration_text,
        [
            r"(?:共|一共|總共|总共|玩|旅行|游玩|行程)\s*([0-9一二两三四五六七八九十]+)\s*(?:天|日)",
            r"([0-9一二两三四五六七八九十]+)\s*(?:天|日)",
            r"玩\s*([0-9一二两三四五六七八九十]+)",
        ],
    )
    travelers = _extract_int(query, [r"([0-9一二两三四五六七八九十]+)\s*(?:个人|人|位)", r"([0-9一二两三四五六七八九十]+)\s*大"])
    budget_text = _trip_budget_text(query)
    budget_values = _budget_values(budget_text)
    start_date, end_date = _extract_travel_dates(query)
    preferences, constraints = _extract_preferences(query)
    if destination and destination not in state.destinations:
        state.destinations.insert(0, destination)
    if origin:
        state.origin = origin
    if duration:
        state.duration_days = duration
    if travelers:
        state.traveler_count = travelers
    if budget_values:
        state.budget.per_person = budget_values if re.search(r"人均|每人|每位", budget_text) else {}
        for field, value in budget_values.items():
            setattr(state.budget, field, value)
    state.budget.normalize_for_travelers(state.traveler_count)
    if start_date:
        state.travel_dates.start_date = start_date
    if end_date:
        state.travel_dates.end_date = end_date
    if not state.duration_days and state.travel_dates.start_date and state.travel_dates.end_date:
        state.duration_days = (state.travel_dates.end_date - state.travel_dates.start_date).days + 1
    for item in preferences:
        if item not in state.preferences.get("items", []):
            state.preferences.setdefault("items", []).append(item)
    for item in constraints:
        if item not in state.hard_constraints:
            state.hard_constraints.append(item)
    if re.search(r"机票|航班|飞机|商务舱|公务舱|头等舱|经济舱|flight", query, re.I):
        flight = state.preferences.setdefault("flight", {})
        flight["request"] = query
        saved = state.preferences.setdefault("flight_preferences", [])
        if query not in saved:
            saved.append(query)
        if re.search(r"预算.{0,8}(?:提高|增加|加|多|高|宽)|(?:提高|增加|加).{0,8}预算|贵一点|多花|上不封顶", query):
            flight["flexible_budget"] = True
        if re.search(r"舒服|舒适|comfortable", query, re.I):
            flight["comfort"] = True
        for cabin in ("头等舱", "商务舱", "公务舱", "超级经济舱", "经济舱"):
            if cabin in query and not re.search(rf"不(?:要|坐|选){cabin}", query):
                flight["cabin"] = cabin
                break
        allowance = _budget_values(query) if _flight_only_request(query) else None
        if allowance:
            flight["budget"] = {**allowance, "basis": "per_person" if re.search(r"人均|每人", query) else "group"}
    return state


def _missing_fields(state: TripState) -> list[str]:
    missing = []
    if not state.origin:
        missing.append("出发城市")
    if not state.duration_days:
        missing.append("旅行天数")
    if not state.traveler_count:
        missing.append("人数")
    if not state.budget.specified:
        missing.append("总预算")
    return missing


def _summarize_trip(trip: TripState) -> str:
    parts = []
    if trip.origin:
        parts.append(f"出发地：{trip.origin}")
    if trip.destination:
        parts.append(f"目的地：{trip.destination}")
    if trip.duration_days:
        parts.append(f"天数：{trip.duration_days}天")
    if trip.traveler_count:
        parts.append(f"人数：{trip.traveler_count}人")
    if trip.budget.amount is not None:
        parts.append(f"全体预算：{trip.budget.amount:g}{trip.budget.currency}")
    elif trip.budget.min_amount is not None:
        ceiling = f"–{trip.budget.max_amount:g}" if trip.budget.max_amount is not None else "起，未设上限"
        parts.append(f"全体预算：{trip.budget.min_amount:g}{ceiling}{trip.budget.currency}")
    elif trip.budget.max_amount is not None:
        parts.append(f"全体预算上限：{trip.budget.max_amount:g}{trip.budget.currency}")
    elif trip.budget.per_person:
        parts.append(f"每人预算：{trip.budget.per_person}（待人数确定后换算全体预算）")
    if trip.preferences.get("items"):
        parts.append("偏好：" + "、".join(trip.preferences["items"]))
    if trip.hard_constraints:
        parts.append("约束：" + "、".join(trip.hard_constraints))
    return "；".join(parts)


def make_nodes(
    database: Database,
    context_builder: ContextBuilder,
    tools: ToolRegistry,
    gateway: ModelGateway,
    renderer: ReportRenderer,
):
    def extract_requirements(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("current_query") or _last_human_text(state.get("messages", []))
        trip = TripState.model_validate(state.get("trip_state") or {})
        trip.user_id = state.get("user_id", trip.user_id)
        trip.trip_id = state.get("trip_id", trip.trip_id)
        trip.thread_id = state.get("thread_id", trip.thread_id)
        trip = _merge_requirements(trip, query)
        trip.current_phase = "intake"
        trip.conversation_summary = _summarize_trip(trip)
        database.save_trip(trip)
        existing_memories = database.list_memories(trip.user_id, trip.trip_id)
        for preference in trip.preferences.get("items", []):
            content = f"用户在旅行规划中偏好：{preference}"
            if not any(item.content == content for item in existing_memories):
                database.save_memory(
                    MemoryItem(
                        user_id=trip.user_id,
                        trip_id=trip.trip_id,
                        kind="preference",
                        content=content,
                        source="user",
                        confidence=1.0,
                        importance=0.75,
                        tags=["travel_preference"],
                        phase="intake",
                    )
                )
        database.append_event(state["thread_id"], trip.user_id, trip.trip_id, "trip.requirements_updated", {"query": query, "state": trip.model_dump(mode="json")})
        return {"current_query": query, "trip_state": trip.model_dump(mode="json"), "step_count": 0}

    def check_missing_fields(state: dict[str, Any]) -> dict[str, Any]:
        trip = TripState.model_validate(state["trip_state"])
        query = state.get("current_query", "")
        wants_discovery = not trip.destination and any(word in query for word in ["推荐", "去哪", "没有目的地", "不知道去哪里", "目的地"])
        travel_words = ["旅行", "旅游", "行程", "目的地", "酒店", "机票", "景点", "路线", "预算", "出发", "玩"]
        has_trip_context = any(
            [
                trip.origin,
                trip.destination,
                trip.duration_days,
                trip.traveler_count,
                trip.budget.amount is not None,
                trip.preferences,
            ]
        )
        if not any(word in query for word in travel_words) and not has_trip_context:
            trip.scenario = "unknown"
            trip.open_questions = []
            database.save_trip(trip)
            return {"trip_state": trip.model_dump(mode="json"), "pending_question": ""}
        missing = _missing_fields(trip)
        if wants_discovery or not trip.destination:
            trip.scenario = "destination_discovery"
        else:
            trip.scenario = "known_destination"
        trip.open_questions = missing
        database.save_trip(trip)
        if missing:
            question = "为了继续规划，我还需要知道：" + "、".join(missing) + "。请尽量一次告诉我。"
            return {"trip_state": trip.model_dump(mode="json"), "pending_question": question}
        return {"trip_state": trip.model_dump(mode="json"), "pending_question": ""}

    def clarify_with_user(state: dict[str, Any]) -> dict[str, Any]:
        question = state.get("pending_question") or "请补充旅行信息。"
        answer = interrupt({"type": "clarification", "question": question})
        trip = TripState.model_validate(state["trip_state"])
        trip = _merge_requirements(trip, str(answer))
        trip.open_questions = _missing_fields(trip)
        database.append_event(state["thread_id"], trip.user_id, trip.trip_id, "user.clarification_received", {"answer": str(answer)})
        return {"messages": [AIMessage(content=question), HumanMessage(content=str(answer))], "current_query": str(answer), "trip_state": trip.model_dump(mode="json"), "pending_question": ""}

    def build_context(state: dict[str, Any]) -> dict[str, Any]:
        trip = TripState.model_validate(state["trip_state"])
        phase = trip.scenario if trip.scenario in {"destination_discovery", "known_destination", "general_planner"} else "intake"
        recent = []
        for message in state.get("messages", [])[-8:]:
            recent.append(f"{message.type}: {message.content}")
        evidence = database.get_evidence(trip.evidence_ids)
        bundle = context_builder.build(
            thread_id=state["thread_id"],
            query=state.get("current_query", ""),
            trip_state=trip,
            phase=phase,
            recent_messages=recent,
            evidence=evidence,
            exposed_tools=tools.names_for_phase(phase),
        )
        return {"phase": phase, "context_text": bundle.text, "context_manifest": bundle.manifest.model_dump(mode="json")}

    def research_destination(state: dict[str, Any]) -> dict[str, Any]:
        trip = TripState.model_validate(state["trip_state"])
        preferences = trip.preferences.get("items", [])
        result = tools.run("search_destinations", {"origin": trip.origin, "duration_days": trip.duration_days, "budget": trip.budget.amount, "preferences": preferences}, state["thread_id"])
        evidence = EvidenceItem(source_name=result.source_name, content=result.summary, structured_data=result.data, related_entities=["destination_discovery"])
        database.save_evidence(evidence)
        trip.evidence_ids.append(evidence.evidence_id)
        if tools.has("web_search"):
            web_result = tools.run(
                "web_search",
                {"query": f"{trip.origin or ''} {trip.duration_days or ''}天旅行目的地推荐 {'、'.join(preferences)}".strip(), "count": 5},
                state["thread_id"],
            )
            web_evidence = EvidenceItem(source_name=web_result.source_name, content=web_result.summary, structured_data=web_result.data, related_entities=["destination_discovery", "web_search"], source_type=web_result.source_type, source_url=web_result.source_url, valid_until=web_result.valid_until)
            database.save_evidence(web_evidence)
            trip.evidence_ids.append(web_evidence.evidence_id)
        trip.current_phase = "destination_discovery"
        trip.destinations = [item["name"] for item in result.data.get("destinations", [])]
        database.save_trip(trip)
        return {"trip_state": trip.model_dump(mode="json"), "tool_results": [result.model_dump(mode="json")], "evidence": [evidence.model_dump(mode="json")]}

    def research_known_destination(state: dict[str, Any]) -> dict[str, Any]:
        trip = TripState.model_validate(state["trip_state"])
        destination = trip.destination or "未知目的地"
        results = []
        evidence_items = []
        calls = [
            ("search_attractions", {"destination": destination, "preferences": trip.preferences.get("items", [])}),
            ("lookup_weather", {"destination": destination}),
            ("search_hotels", {"destination": destination, "duration_days": trip.duration_days}),
            ("search_flights", {"origin": trip.origin, "destination": destination}),
        ]
        for name, arguments in calls:
            result = tools.run(name, arguments, state["thread_id"])
            results.append(result.model_dump(mode="json"))
            evidence = EvidenceItem(source_name=result.source_name, content=result.summary, structured_data=result.data, related_entities=["known_destination", name])
            database.save_evidence(evidence)
            evidence_items.append(evidence)
            trip.evidence_ids.append(evidence.evidence_id)
        if tools.has("web_search"):
            web_result = tools.run(
                "web_search",
                {"query": f"{destination} 旅行景点攻略 酒店交通 注意事项", "count": 5},
                state["thread_id"],
            )
            web_evidence = EvidenceItem(source_name=web_result.source_name, content=web_result.summary, structured_data=web_result.data, related_entities=["known_destination", "web_search"], source_type=web_result.source_type, source_url=web_result.source_url, valid_until=web_result.valid_until)
            database.save_evidence(web_evidence)
            evidence_items.append(web_evidence)
            trip.evidence_ids.append(web_evidence.evidence_id)
        attractions = next((item["data"].get("attractions", []) for item in results if item["tool_name"] == "search_attractions"), [])
        hotels = next((item["data"].get("hotels", []) for item in results if item["tool_name"] == "search_hotels"), [])
        flights = next((item["data"].get("options", []) for item in results if item["tool_name"] == "search_flights"), [])
        route = tools.run("plan_route", {"places": attractions, "duration_days": trip.duration_days}, state["thread_id"])
        results.append(route.model_dump(mode="json"))
        route_evidence = EvidenceItem(source_name=route.source_name, content=route.summary, structured_data=route.data, related_entities=["known_destination", "plan_route"])
        database.save_evidence(route_evidence)
        evidence_items.append(route_evidence)
        trip.evidence_ids.append(route_evidence.evidence_id)
        trip.selected_places = attractions
        trip.hotel_options = hotels
        trip.transport_options = flights
        trip.route_plan = route.data.get("route", [])
        trip.current_phase = "known_destination"
        database.save_trip(trip)
        return {"trip_state": trip.model_dump(mode="json"), "tool_results": results, "evidence": [item.model_dump(mode="json") for item in evidence_items]}

    def general_planner(state: dict[str, Any]) -> dict[str, Any]:
        trip = TripState.model_validate(state["trip_state"])
        trip.scenario = "general_planner"
        trip.current_phase = "general_planner"
        trip.open_questions.append("这是自由规划请求，当前 MVP 将使用旅行规划工具的有限组合处理。")
        return {"trip_state": trip.model_dump(mode="json"), "error": "general_planner_fallback"}

    def build_plan(state: dict[str, Any]) -> dict[str, Any]:
        trip = TripState.model_validate(state["trip_state"])
        results = state.get("tool_results", [])
        if trip.scenario == "destination_discovery":
            candidates = next((item.get("data", {}).get("destinations", []) for item in results if item.get("tool_name") == "search_destinations"), [])
            destination = candidates[0]["name"] if candidates else None
            summary = "根据预算、时间和偏好生成了目的地候选。建议先确认目的地，再继续生成详细行程。"
            itinerary = [ItineraryDay(day=1, title="目的地比较与初步安排", activities=candidates[:3], estimated_cost=0)]
            plan = TravelPlan(trip_id=trip.trip_id, title="目的地推荐方案", summary=summary, destination=destination, assumptions=["候选信息当前来自 Mock 工具，需接入真实服务后再次核验"], itinerary=itinerary, evidence_ids=trip.evidence_ids, warnings=["当前为目的地候选阶段，不代表已完成预订。"])
        else:
            budget_result = tools.run("calculate_budget", {"duration_days": trip.duration_days, "traveler_count": trip.traveler_count, "hotel_nightly_price": (trip.hotel_options[0].get("nightly_price", 420) if trip.hotel_options else 420), "transport_price": (trip.transport_options[0].get("estimated_price", 900) if trip.transport_options else 900)}, state["thread_id"])
            route = trip.route_plan or [{"day": 1, "title": "城市核心体验", "places": trip.selected_places[:2]}]
            itinerary = [ItineraryDay(day=item.get("day", index + 1), title=item.get("title", "旅行安排"), activities=item.get("places", []), estimated_cost=0) for index, item in enumerate(route)]
            trip.budget_breakdown = budget_result.data.get("breakdown", {})
            plan = TravelPlan(trip_id=trip.trip_id, title=f"{trip.destination or '旅行'}行程规划", summary=f"这是一个面向 {trip.traveler_count} 位旅行者、约 {trip.duration_days} 天的初步方案。", destination=trip.destination, assumptions=["酒店、交通和天气信息当前包含 Mock 数据，出行前必须重新核验"], itinerary=itinerary, hotel_options=trip.hotel_options, transport_options=trip.transport_options, budget_breakdown=trip.budget_breakdown, total_estimated_cost=float(budget_result.data.get("total", 0)), evidence_ids=trip.evidence_ids, warnings=["价格、库存、营业时间和交通时刻未执行真实预订核验。"])
        trip.report_version += 1
        database.save_trip(trip)
        return {"trip_state": trip.model_dump(mode="json"), "plan": plan.model_dump(mode="json")}

    def render_report(state: dict[str, Any]) -> dict[str, Any]:
        trip = TripState.model_validate(state["trip_state"])
        plan = TravelPlan.model_validate(state["plan"])
        markdown_path, html_path = renderer.render(plan)
        recent = [f"{message.type}: {message.content}" for message in state.get("messages", [])[-8:]]
        final_context = context_builder.build(
            thread_id=state["thread_id"],
            query=state.get("current_query", ""),
            trip_state=trip,
            phase=trip.current_phase,
            recent_messages=recent,
            evidence=database.get_evidence(trip.evidence_ids),
            exposed_tools=tools.names_for_phase(trip.current_phase),
        )
        fallback = f"已生成旅行规划：{plan.title}\nMarkdown：{markdown_path}\nHTML：{html_path}\n\n{plan.summary}"
        response = gateway.generate(
            "你是旅行规划助手。用中文简洁地向用户总结已经生成的旅行计划。不要声称已经预订或获得实时价格。",
            f"上下文摘要：\n{final_context.text}\n\n旅行计划：{plan.model_dump_json(indent=2)}",
            fallback,
        )
        database.append_event(state["thread_id"], trip.user_id, trip.trip_id, "report.generated", {"markdown_path": str(markdown_path), "html_path": str(html_path), "plan": plan.model_dump(mode="json")})
        return {"assistant_response": response, "messages": [AIMessage(content=response)], "trip_state": trip.model_dump(mode="json"), "context_text": final_context.text, "context_manifest": final_context.manifest.model_dump(mode="json")}

    def safe_stop(state: dict[str, Any]) -> dict[str, Any]:
        message = "当前请求超出第一版旅行工具范围，已安全停止。你可以补充目的地、日期、预算，或稍后接入对应的真实服务。"
        return {"assistant_response": message, "messages": [AIMessage(content=message)]}

    return {
        "extract_requirements": extract_requirements,
        "check_missing_fields": check_missing_fields,
        "clarify_with_user": clarify_with_user,
        "build_context": build_context,
        "research_destination": research_destination,
        "research_known_destination": research_known_destination,
        "general_planner": general_planner,
        "build_plan": build_plan,
        "render_report": render_report,
        "safe_stop": safe_stop,
    }
