"""Small, task-independent model views. Full research stays in the evidence store."""
from __future__ import annotations

import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from travel_agent.context.builder import _context_safe_data
from travel_agent.tools.normalization import flight_elapsed_minutes


class ContextCapacityError(ValueError):
    """Request semantic compaction or a smaller set of bound tool definitions."""


def dump(value):
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def conservative_tokens(value):
    # UTF-8 bytes provide a conservative fallback for Chinese and mixed JSON.
    # The provider's actual usage remains authoritative in the model trace.
    return max(1, math.ceil(len(str(value).encode("utf-8")) / 2))


def project(value, *, depth=0, limit=8, text_limit=900):
    """Bound containers without cutting JSON or leaking opaque provider credentials."""
    value = _context_safe_data(value)
    if isinstance(value, str):
        if len(value) <= text_limit:
            return value
        boundary = value.rfind("\n", 0, text_limit)
        return value[:boundary if boundary > text_limit // 2 else text_limit] + "…（仅预览，完整内容可按需读取）"
    if isinstance(value, list):
        rows = [project(x, depth=depth + 1, limit=limit, text_limit=text_limit) for x in value[:limit]]
        if len(value) > limit:
            rows.append({"remaining_items": len(value) - limit})
        return rows
    if isinstance(value, dict):
        omitted = {"images", "image_url", "image_urls", "photos", "raw", "html", "cookie"}
        return {
            k: project(v, depth=depth + 1, limit=limit, text_limit=text_limit)
            for k, v in value.items()
            if k not in omitted and (depth < 8 or not isinstance(v, (dict, list)))
        }
    return value


def tool_view(value, limit=5, text_limit=650):
    """A bounded overview, with complete price records and retrievable provenance."""
    from travel_agent.tools.normalization import rail_text_records
    trains = rail_text_records(value) if isinstance(value, dict) else []
    if not trains and isinstance(value, dict) and isinstance(value.get("items"), list):
        for row in value["items"]:
            if isinstance(row, dict) and row.get("start_train_code"):
                trains.append({
                    "train": row.get("start_train_code"), "origin": row.get("from_station"),
                    "destination": row.get("to_station"), "departure_time": row.get("start_time"),
                    "arrival_time": row.get("arrive_time"), "duration": row.get("lishi"),
                    "seats": {p.get("seat_name"): {"availability": p.get("num"), "price_cny_per_person": p.get("price")}
                              for p in row.get("prices", []) if p.get("seat_name") in {"二等座", "一等座"}},
                })
    if trains:
        return {"train_count": len(trains), "trains": trains[:limit],
                "preview_only": len(trains) > limit}
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        rows = value["results"]
        return {"results": [
            {"title": r.get("title"), "url": r.get("url"),
             "content": str(r.get("description") or r.get("snippet") or r.get("text") or r.get("content") or "")[:text_limit]}
            for r in rows[:min(limit, 3)] if isinstance(r, dict)
        ], "result_count": len(rows), "preview_only": len(rows) > min(limit, 3)}
    preview = project(value, limit=limit, text_limit=550)
    # Large heterogeneous providers can have many small fields; also cap total size.
    while len(dump(preview)) > 6500 and limit > 1:
        limit -= 1
        preview = project(value, limit=limit, text_limit=400)
    if len(dump(preview)) > 6500:
        return {"preview_only": True, "summary": dump(preview)[:5000], "next": "按证据ID读取必要字段"}
    return preview


def selection_view(name, data):
    """Provide small complete comparison cards while retaining every raw quote."""
    if name == "amap_maps__maps_weather" and isinstance(data.get("forecasts"), list):
        weekdays = {"1": "周一", "2": "周二", "3": "周三", "4": "周四", "5": "周五", "6": "周六", "7": "周日"}
        return {"city": data.get("city"), "forecasts": [
            {"date": item.get("date"), "weekday": weekdays.get(str(item.get("week"))),
             **{k: item[k] for k in ("dayweather", "nightweather", "daytemp", "nighttemp", "daywind", "daypower") if k in item}}
            for item in data["forecasts"] if isinstance(item, dict)
        ]}
    if name != "google_flights__search_flights" or str(data.get("trip_type")).lower() != "one_way":
        return tool_view(data)
    raw = data.get("flights", [])
    options = []
    for index, item in enumerate(raw):
        legs, stops = item.get("legs", []), item.get("layovers", [])
        if not legs or not isinstance(item.get("price"), int | float):
            continue
        elapsed = flight_elapsed_minutes(item)
        options.append({
            "item_index": index, "price": item["price"], "currency": item.get("currency"),
            "airline": item.get("primary_airline_name"),
            "departs_local": legs[0].get("departure_time"), "arrives_local": legs[-1].get("arrival_time"),
            "total_minutes_including_connections": elapsed,
            "total_duration_label": f"{int(elapsed) // 60}小时{int(elapsed) % 60:02d}分" if elapsed is not None else None,
            "flight_numbers": [leg.get("flight_number") for leg in legs],
            "connections": [{"airport": s.get("airport"), "minutes": s.get("duration")} for s in stops],
        })
    if not options:
        return tool_view(data)
    eligible = [x for x in options if x["total_minutes_including_connections"] is not None]
    by_price = sorted(options, key=lambda x: x["price"])
    fastest = min(eligible, key=lambda x: x["total_minutes_including_connections"]) if eligible else None
    fastest_by_price = []
    duration = math.inf
    for option in sorted(eligible, key=lambda x: (x["price"], x["total_minutes_including_connections"])):
        if option["total_minutes_including_connections"] < duration:
            fastest_by_price.append(option)
            duration = option["total_minutes_including_connections"]
    shortlist = list({x["item_index"]: x for x in [*by_price[:2], *fastest_by_price[:4],
                                                     *([fastest] if fastest else [])]}.values())
    return {"candidate_count": len(raw), "trip_type": "one_way", "quote_context": data.get("quote_context"),
            "cheapest_index": by_price[0]["item_index"], "fastest_index": fastest["item_index"] if fastest else None,
            "shortlist": shortlist, "next": "用现成候选比较价格、总耗时和中转；以total_duration_label为准。仅在用户确实需要额外字段时读取原始记录。"}


def recent_lookup_quotes(database, thread_id):
    """Restore small quote cards for a same-query follow-up, without restoring the transcript."""
    from travel_agent.models import ToolResult
    from travel_agent.persistence.runtime import RuntimeStore

    cards = []
    seen = set()
    name = "google_flights__search_flights"
    fields = ("origin", "destination", "departure_date", "return_date", "passengers", "cabin_class", "currency")
    for item in database.runtime.index(thread_id, limit=8, tool_name=name):
        if not item["success"]:
            continue
        row = database.runtime.result(thread_id, item["result_id"])
        if not row:
            continue
        result = ToolResult.model_validate(row["result"])
        if not RuntimeStore.reusable(name, result):
            continue
        args = item["arguments"]
        key = tuple(args.get(k) for k in fields)
        if key in seen:
            continue
        view = selection_view(name, result.data)
        if "shortlist" not in view:
            continue
        seen.add(key)
        cards.append({"result_id": item["result_id"], "fetched_at": item["fetched_at"],
                      "arguments": dict(zip(fields, key)), "data": view})
        if len(cards) == 3:
            break
    return cards


def compact_tool_message(message, char_limit=6500):
    try:
        payload = json.loads(str(message.content))
    except (TypeError, ValueError):
        payload = {"content": str(message.content)}
    if payload.get("prepared_view") and len(str(message.content)) <= char_limit:
        return message
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = {**payload, "data": tool_view(payload["data"])}
    view = project(payload, limit=5, text_limit=750)
    for limit, text_limit in [(3, 450), (2, 240), (1, 120)]:
        if len(dump(view)) <= char_limit:
            break
        view = project(payload, limit=limit, text_limit=text_limit)
    if len(dump(view)) > char_limit:
        view = {
            "success": payload.get("success") if isinstance(payload, dict) else None,
            "evidence_id": payload.get("evidence_id") if isinstance(payload, dict) else None,
            "preview_only": True, "next": "完整结果已保存；用read_state按证据ID读取必要部分",
        }
    return ToolMessage(content=dump(view), tool_call_id=message.tool_call_id, name=message.name)


def trip_summary(trip, report=None, *, include_details=True):
    facts = trip.to_context_dict()
    for key in (
        "selected_places", "hotel_options", "transport_options", "route_plan",
        "budget_breakdown", "evidence_ids",
    ):
        facts.pop(key, None)
    if report:
        saved = json.loads(report["payload_json"]).get("report", {})
        facts["saved_report"] = {
            "version": report["version"],
            "html_path": report["html_path"],
            "markdown_path": report["markdown_path"],
            "quality": saved.get("quality"),
            "hotels": [{"city": g.get("city"), "checkin": g.get("checkin"), "checkout": g.get("checkout"),
                        "candidates": [{**{k: h.get(k) for k in ("name", "baseline", "total_price", "currency")},
                                        "url": str(h.get("url") or "").split("?", 1)[0]}
                                       for h in g.get("hotels", [])]}
                       for g in saved.get("hotel_regions", [])],
            "budget": {k: saved.get("budget", {}).get(k) for k in ("low", "high", "currency")},
            "day_index": [
                {"day": d["day"], "date": d.get("date"), "city": d.get("city"),
                 "places": [p["name"] for p in d.get("stops", [])],
                 "activities": [{k: p.get(k) for k in ("name", "start_time", "duration_minutes")}
                                for p in d.get("stops", [])],
                 "type": d.get("type"), "timing_warning": d.get("timing_warning")}
                for d in saved.get("days", [])
            ],
        }
        if not include_details:
            facts["saved_report"] = {k: facts["saved_report"][k] for k in ("version", "html_path", "markdown_path", "quality", "budget")}
    facts["available_records"] = {
        "places": len(trip.selected_places), "hotels": len(trip.hotel_options),
        "transport": len(trip.transport_options), "days": len(trip.route_plan),
        "evidence": len(trip.evidence_ids),
    }
    if report:
        facts.pop("available_records", None)
    return facts


def transfer_index(edge):
    option = next((o for o in edge.get("options", []) if o.get("baseline")), next(iter(edge.get("options", [])), {}))
    selected = {k: option.get(k) for k in ("mode", "status", "duration_minutes", "planning_duration_minutes") if option.get(k) is not None}
    for key in ("departure_date", "departure_time", "arrival_date", "arrival_time"):
        selected[key] = option.get("planning_" + key) or option.get(key) or (edge.get("date") if key == "departure_date" else None)
    return {**{k: edge.get(k) for k in ("id", "origin", "destination", "scope")}, "date": selected.get("departure_date"), "selected": selected,
            "time_basis": "当前生效的计划时刻；原始班次与查询日期在来源记录中保留"}


def build_agent_messages(database, settings, state, policy, catalog, skills, definitions):
    from travel_agent.reports.route_contract import approved_plan_view, reserve_kinds

    trip = state["_trip"]
    scope = state.get("request_scope", {})
    advisory = scope.get("scenario") in {"answer", "compare", "lookup"}
    from travel_agent.memory import MemoryRepository

    memory_repository = MemoryRepository(database)
    memory_items = memory_repository.candidates(
        trip.user_id, trip.trip_id, state.get("current_query", ""), scope.get("scenario")
    )[:6]
    episodes = memory_repository.episodes(trip.user_id, limit=3)
    sections = {
        "runtime": {
            "current_date": datetime.now(ZoneInfo("Asia/Hong_Kong")).date().isoformat(),
            "timezone": "Asia/Hong_Kong",
        },
        "current_request": state["current_query"],
        "request_scope": scope,
        "active_goal": state.get("active_goal"),
        "root_goal": state.get("root_goal"),
        "task_plan": state.get("task_plan", []),
        "selected_results": [{k: s[k] for k in ("result_id", "item_index")} for s in state.get("selected_results", [])],
        "run_status": state.get("run_status", "running"),
        "semantic_summary": state.get("semantic_summary", {}),
        "trip": trip_summary(trip, database.latest_report(trip.trip_id), include_details=not bool(state.get("report_workspace"))),
        "capabilities": catalog,
        "skills": [{"name": s["name"], "description": s["description"]} for s in skills.values()],
        "memory": [
            {"kind": m.kind, "key": m.key, "content": m.content[:240], "source": m.source,
             "origin": "user" if m.source == "user" else m.source}
            for m in memory_items
        ],
    }
    if episodes:
        sections["recent_trips"] = [
            {
                "trip": item.structured_value.get("title"),
                "dates": [item.structured_value.get("start_date"), item.structured_value.get("end_date")],
                "travelers": item.structured_value.get("travelers"),
                "goals": (item.structured_value.get("goals") or [])[:2],
                "decisions": (item.structured_value.get("decisions") or [])[:3],
                "open_items": (item.structured_value.get("open_items") or [])[:3],
                "status": item.structured_value.get("status"),
            }
            for item in episodes
        ]
    if advisory:
        for key in ("active_goal", "root_goal", "task_plan", "semantic_summary"):
            sections.pop(key, None)
    if approved := approved_plan_view(state):
        sections["approved_plan"] = approved
    if scope.get("scenario") == "plan_review":
        if pending := state.get("pending_plan"):
            sections["previous_proposal"] = {"version": pending["version"], "scope": pending["scope"],
                                             "text": pending["proposal"][:2400]}
        if proposal := state.get("unpresented_plan"):
            sections["revision_requests"] = proposal.get("requests") or [proposal["request"]]
    if material := state.get("delivery_material"):
        sections["completed_step_material"] = material
    if quotes := state.get("recent_lookup_quotes"):
        sections["matching_previous_quotes"] = quotes
        sections["quote_reuse"] = (
            "本轮在已有查询上调整选择条件。保存的报价与对象、人数、日期一致时直接使用。"
            "只给用户问的推荐和取舍，需要对比时每个出发地挑一个相关的候选。"
            "采用报价卡上的日期、当地时刻、总耗时与中转信息；未提供的订票条件不作保证。"
        )
    cursors = [entry["next_args"] for fields in reversed(state.get("read_progress", {}).values()) for entry in fields.values()
               if entry.get("next_args")]
    if cursors:
        sections["unread_pages"] = cursors[:8]
    sections["result_index"] = project(database.runtime.index(state["thread_id"], limit=8), limit=8, text_limit=200)
    # Keep short source content when old tool-message groups are compacted.
    # A list of call names alone is not enough for grounded synthesis.
    source_cards = []
    for result in state.get("tool_results", [])[-6:]:
        if not result.get("success"):
            continue
        data = result.get("data", {})
        if isinstance(data.get("results"), list):
            source_cards.append({
                "tool": result.get("tool_name"),
                "sources": [
                    {"title": r.get("title"), "url": r.get("url"),
                     "excerpt": str(r.get("description") or r.get("snippet") or "")[:450]}
                    for r in data["results"][:2]
                ],
            })
        elif isinstance(data.get("feeds"), list):
            source_cards.append({
                "tool": result.get("tool_name"), "kind": "搜索标题，不等于正文",
                "notes": [
                    {"id": f.get("id"), "title": (f.get("noteCard") or {}).get("displayTitle")}
                    for f in data["feeds"][:3]
                ],
            })
        elif (
            isinstance(data.get("data"), dict)
            and isinstance(data["data"].get("note"), dict)
            and data["data"]["note"].get("desc")
        ):
            note = data["data"]["note"]
            source_cards.append({"title": note.get("title"), "excerpt": note["desc"][:900]})
    exchanges_size = sum(
        conservative_tokens(str(m.content)) for m in state.get("agent_messages", [])
    )
    if source_cards and exchanges_size > settings.context_token_budget // 3:
        sections["retrieved_source_cards"] = source_cards
    workspace = state.get("report_workspace", {})
    if workspace and not advisory:
        data = workspace.get("data", {})
        sections["report_workspace"] = {
            "authority": "这是当前整理的成果。已有文件是上次保存的快照；复用现有结果，按本轮批准的方式交付。",
            "request": workspace.get("request"), "sections": workspace.get("sections"),
            "completed": workspace.get("completed"), "rendered": workspace.get("rendered", False),
            "cities": [
                {"id": c["id"], "name": c["name"], "days": c["days"],
                 "visit_days": c["days"] - len(reserve_kinds(c)),
                 "places": sum(p.get("city_id") == c["id"] for p in data.get("places", []))}
                for c in data.get("region", {}).get("cities", [])
            ],
            "transport_count": len(data.get("transport", [])),
            "transfer_index": [transfer_index(edge) for edge in data.get("transport", [])],
            "flight_quotes": [{**{k: option.get(k) for k in ("per_person_price", "group_estimated_price", "quote_passengers", "requested_passengers", "currency", "source_result_id")},
                               "directions": [{k: d.get(k) for k in ("direction", "origin", "destination", "departure_time", "arrival_time", "flight_numbers")} for d in option["directions"]]}
                              for edge in data.get("transport", []) for option in edge.get("options", []) if option.get("baseline") and option.get("directions")],
            "stay_index": [{"city": g.get("city"), "checkin": g.get("checkin"), "checkout": g.get("checkout"), "nights": g.get("nights"),
                            "hotels": [{k: h.get(k) for k in ("name", "baseline", "total_price", "currency")} for h in g.get("hotels", [])]}
                           for g in data.get("hotel_regions", [])],
            "days_count": len(data.get("days", [])),
            "daily_index": [{"day": d["day"], "city": d.get("city"), "stay_city": d.get("stay_city"), "type": d.get("type"),
                             "stops": [{"id": p["id"], "name": p["name"]} for p in d.get("stops", [])],
                             "timing_warning": d.get("timing_warning"), "research_gap": d.get("research_gap")}
                            for d in data.get("days", [])],
            "experience_index": [{"id": p["id"], "city": p.get("city"), "name": p["name"], "kind": p.get("kind"),
                                  "selected": p["id"] in data.get("selected_place_ids", [])} for p in data.get("places", [])],
            "stay_count": len(data.get("hotel_regions", [])),
            "issues": workspace.get("active_issues", []),
            "research_progress": {
                name: {k: value.get(k) for k in ("phase", "completed", "places")}
                for name, value in workspace.get("research_progress", {}).items()
            },
        }
    # Stable capability/skill metadata precedes state that changes after each call.
    stable_keys = tuple(k for k in ("runtime", "capabilities", "skills", "memory", "recent_trips") if k in sections)
    sections = {**{k: sections[k] for k in stable_keys}, **{k: v for k, v in sections.items() if k not in stable_keys}}
    # Reserve the actual bound schemas; never silently drop the capability catalog.
    schema_tokens = conservative_tokens(dump(definitions))
    skill_text = "\n".join(state.get("loaded_skills", {}).values())
    used = (
        conservative_tokens(policy) + conservative_tokens(dump(sections))
        + schema_tokens + conservative_tokens(skill_text)
    )
    available = settings.context_token_budget - 2400 - used
    if available < 0:
        raise ContextCapacityError("能力定义与当前请求超过本轮上下文预算，需要缩小已加载能力范围。")
    exchanges = [
        compact_tool_message(m) if isinstance(m, ToolMessage)
        else AIMessage(content=m.content, tool_calls=m.tool_calls) if isinstance(m, AIMessage)
        else m for m in state.get("agent_messages", [])
    ]
    # Semantic compaction happens before this function; never replace facts with call names.
    cost = sum(conservative_tokens(dump(m.model_dump())) for m in exchanges)
    compacted_exchanges = False
    if cost > available:
        count = max(1, sum(isinstance(m, ToolMessage) for m in exchanges))
        per_message = max(400, int(available / count / 2))
        exchanges = [
            compact_tool_message(m, char_limit=per_message) if isinstance(m, ToolMessage) else m
            for m in exchanges
        ]
        cost = sum(conservative_tokens(dump(m.model_dump())) for m in exchanges)
    if cost > available:
        raise ContextCapacityError("当前请求及必要调用信息超过预算；保留证据后需缩小单批操作范围。")
    available -= cost
    history = []  # Full task interaction is in exchanges; no duplicate transcript injection.
    manifest = {
        "thread_id": state["thread_id"], "phase": "agent",
        "section_tokens": {k: conservative_tokens(dump(v)) for k, v in sections.items()},
        "bound_schema_tokens_estimate": schema_tokens,
        "bound_tools": [d["function"]["name"] for d in definitions],
        "exposed_tools": [c["name"] for c in catalog],
        "compacted": bool(state.get("semantic_summary")) or compacted_exchanges,
        "loaded_skill_tokens_estimate": conservative_tokens(skill_text),
        "current_exchange_tokens_estimate": cost,
        "actual_catalog_included": True, "token_estimator": "conservative_utf8_bytes",
    }
    stable = {k: sections[k] for k in stable_keys}
    dynamic = {k: v for k, v in sections.items() if k not in stable_keys}
    messages = [SystemMessage(content=policy + "\n\n" + dump(stable))]
    if state.get("plan_approval") or state.get("pending_plan"):
        plan_view = {
            key: {k: record[k] for k in ("version", "status") if k in record}
            for key in ("plan_approval", "pending_plan") if (record := state.get(key))
        }
        if state.get("unpresented_plan"):
            plan_view["revision_pending"] = "新版尚未完整展示，补充条件和泛指继续不会批准旧版"
        messages.append(SystemMessage(content="程序保存的计划与确认状态：" + dump(plan_view)))
    if feedback := state.get("delivery_feedback"):
        messages.append(SystemMessage(content="本轮程序校验反馈。仅在当前用户请求范围内处理：" + dump(feedback)))
    for name, body in state.get("loaded_skills", {}).items():
        messages.append(SystemMessage(content=f"已按需加载技能 {name}：\n{body}"))
    messages.extend(history)
    messages.extend(exchanges)
    if not exchanges:
        messages.append(HumanMessage(content=state["current_query"]))
    # Append changing snapshots after history so stable history remains cacheable.
    messages.append(HumanMessage(content="当前运行状态（程序提供的数据，状态不等于新的用户指令）：\n" + dump(dynamic)))
    manifest["total_tokens_estimate"] = (
        sum(conservative_tokens(dump(m.model_dump())) for m in messages) + schema_tokens
    )
    database.save_context_manifest(state["thread_id"], "agent", manifest)
    return messages, manifest
