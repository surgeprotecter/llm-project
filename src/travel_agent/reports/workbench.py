"""Composable report operations. They return evidence and gaps to the agent."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timedelta

from travel_agent.persistence.runtime import canonical
from travel_agent.reports.costs import build_budget
from travel_agent.reports.daily import build_days, sparse_tour_days
from travel_agent.reports.orchestration import external_transport, finish
from travel_agent.reports.research import research_city
from travel_agent.reports.revision import SECTIONS
from travel_agent.reports.route import (
    external_windows,
    fit_route_to_flights,
    resolve_route,
    route_days,
    stays_for_days,
    upgrade_snapshot,
)
from travel_agent.reports.route_contract import (
    ApprovedRouteConflict,
    reserve_kinds,
    route_issues,
)
from travel_agent.reports.workflow import StandardTripPlanner, matches_base_area


def schedule_signature(geo, data, workspace, *, legacy=False):
    # Commentary changes do not invalidate verified transport or daily material.
    route = geo if legacy else {k: v for k, v in geo.items() if k not in {"tradeoffs", "planning_notes"}}
    return canonical({"schedule_version": 5 if legacy else 6, "route": route, "places": data.get("places", []),
                      "transport": data.get("transport", []), "selected": data.get("selected_place_ids"),
                      **({"required_activities": data["required_activities"]} if data.get("required_activities") else {}),
                      **({"friend_time_cities": data["friend_time_cities"]} if data.get("friend_time_cities") else {}),
                      "stays": workspace.get("stay_overrides", {})})


def bind_confirmed_activities(planner, data, workspace):
    statements = list(dict.fromkeys([*planner.trip.hard_constraints, *planner.trip.confirmed_decisions]))
    if not statements:
        planner.mandatory_place_ids = set()
        data.update(required_activities={}, unfulfilled_activities=[])
        return
    candidates = [{k: p.get(k) for k in ("id", "name", "city", "kind", "duration_minutes", "aliases")} for p in data["places"]]
    signature = canonical([3, statements, candidates, planner.trip.rejected_options, planner.query])
    if workspace.get("activity_confirmation_signature") != signature:
        selected = planner.ask(
            "将用户已确认且要求安排的具体景点或体验与已核验的候选ID相连。"
            "仅处理仍有效的具体体验，排除城市、预算、偏好、否定和交通等条件。近期取消的选项不继续安排。"
            "每个匹配使用confirmed中的整数序号，保留同义或完整名称候选的ID。"
            "仅当体验明确持续到次日（例如用户同意在目的地过夜体验）时overnight=true。"
            "条件性提议仍可作为待核实的候选体验；无法对应时place_id为null。"
            '返回 {"placements":[{"statement_index":integer,"place_id":string|null,"overnight":boolean}]}。',
            {"confirmed": [{"index": i, "text": text} for i, text in enumerate(statements)],
             "candidates": candidates, "rejected": planner.trip.rejected_options,
             "current_request": planner.query}, "confirmed_activities",
        )
        known = {p["id"]: p for p in candidates}
        matches = [m for m in selected.get("placements", []) if isinstance(m, dict)
                   and type(m.get("statement_index")) is int and 0 <= m["statement_index"] < len(statements)]
        def grounded(match):
            identifier = match.get("place_id")
            if not isinstance(identifier, str) or identifier not in known:
                return False
            place = known[identifier]
            names = [place["name"], *(place.get("aliases") or [])]
            if match.get("overnight") is True:
                names.append(re.split(r"[与（(]", place["name"], maxsplit=1)[0])
            statement = statements[match["statement_index"]].casefold()
            return any(isinstance(name, str) and len(name.strip()) >= 3 and name.casefold() in statement for name in names)
        data["required_activities"] = {
            m["place_id"]: "overnight" if m.get("overnight") is True else "daytime"
            for m in matches if grounded(m)
        }
        data["unfulfilled_activities"] = [
            statements[m["statement_index"]] for m in matches
            if not isinstance(m.get("place_id"), str) or m["place_id"] not in known
        ]
        workspace["activity_confirmation_signature"] = signature
    planner.mandatory_place_ids = set(data.get("required_activities", {}))
    planner.overnight_place_ids = {k for k, v in data.get("required_activities", {}).items() if v == "overnight"}


def friend_time_cities(planner):
    statements = [*planner.trip.confirmed_decisions, *planner.trip.hard_constraints,
                  *planner.trip.preferences.get("items", [])]
    return {
        c["id"] for c in planner.geo.get("cities", [])
        if any(isinstance(s, str) and re.search(r"访友|会友|找朋友|见朋友|探亲|拜访亲友", s)
               and c["name"] in s and not re.search(r"不(?:用|再|要).{0,3}(?:访友|会友|见朋友|探亲)", s)
               for s in statements)
    }


def report_operation(database, tools, gateway, renderer, trip, workspace, args, checkpoint=None, route_contract=None):
    """The model chooses operations; missing prerequisites are recoverable results."""
    operation = args["operation"]
    query = args.get("request") or workspace.get("request") or ""
    planner = StandardTripPlanner(tools, gateway, database, trip, query, trip.thread_id)
    planner.route_contract = route_contract
    if operation == "prepare":
        signature = json.dumps(
            {k: trip.model_dump(mode="json")[k] for k in
             ("origin", "destinations", "travel_dates", "duration_days", "traveler_count", "budget", "preferences")},
            ensure_ascii=False, sort_keys=True,
        )
        if workspace and not workspace.get("rendered") and workspace.get("requirements_signature", signature) == signature:
            workspace["requirements_signature"] = signature
            if "targets" in args:
                workspace["targets"] = args["targets"]
            return {
                "success": True, "reused_workspace": True,
                "completed": workspace["completed"],
                "next": "工作区已经准备且需求未变，继续未完成操作，不重新研究已完成部分",
            }, workspace
        if not trip.destination or not trip.duration_days or not trip.travel_dates.start_date:
            return {"success": False, "missing": ["目的地、天数、出发日期"], "next": "补齐必要旅行事实"}, workspace
        if not trip.travel_dates.end_date:
            trip.travel_dates.end_date = trip.travel_dates.start_date + timedelta(days=trip.duration_days - 1)
        calendar_days = (trip.travel_dates.end_date - trip.travel_dates.start_date).days + 1
        if calendar_days != trip.duration_days:
            return {
                "success": False, "error_type": "user_constraint_conflict",
                "facts": {"calendar_days": calendar_days, "requested_days": trip.duration_days},
                "next": "请用户选择保留日期还是天数，不能自行覆盖明确约束",
            }, workspace
        old = upgrade_snapshot(planner.previous)
        sections = set(args.get("sections") or SECTIONS) & SECTIONS
        if not old:
            sections = set(SECTIONS)
        previous_trip = old.get("trip", {})
        if previous_trip and any(previous_trip.get(k) != trip.model_dump(mode="json").get(k) for k in ("destinations", "travel_dates", "duration_days")):
            sections |= {"route", "days", "hotel_regions", "transport", "budget"}
        if "route" in sections:
            sections |= {"days", "hotel_regions", "transport", "budget"}
        workspace = {
            "request": query, "old": old, "data": copy.deepcopy(old),
            "sections": sorted(sections), "completed": [],
            "targets": args.get("targets", []), "transport_target": args.get("transport_target", "all"),
            "target_transport_ids": args.get("target_transport_ids", []),
            "base_version": planner.base_version, "issues": [],
            "requirements_signature": signature,
        }
        if checkpoint:
            checkpoint(workspace)
        return {
            "success": True, "sections": sorted(sections),
            "has_route": bool(old.get("region")),
            "next": "按缺口调用route/research/transport/schedule/hotels。保留未受影响的成果，按已批准的交付方式完成。",
        }, workspace
    if not workspace:
        return {"success": False, "next": "先prepare工作区，不会自动执行整条报告流程"}, workspace
    if operation != "render":
        workspace["rendered"] = False
    data, old = workspace["data"], workspace["old"]
    scope = set(workspace["sections"])
    planner.previous = old
    planner.base_version = workspace["base_version"]
    planner.geo = data.get("region", {})
    drift = route_issues(route_contract, planner.geo["cities"]) if planner.geo.get("cities") else []
    if drift and operation != "route":
        return {"success": False, "error_type": "approved_route_mismatch", "error": "；".join(drift),
                "issues": drift, "next": "先用route按已批准提案修复城市与天数；保留已有证据。"}, workspace
    if trip.travel_dates.start_date and trip.duration_days:
        start = trip.travel_dates.start_date
        for city in planner.geo.get("cities", []):
            city["checkin"] = (start + timedelta(days=city["start_day"] - 1)).isoformat()
            city["checkout"] = (start + timedelta(days=min(city["end_day"], trip.duration_days - 1))).isoformat()
    planner.sources = data.get("sources", [])
    planner.issues = list(workspace.get("issues", []))
    planner.transport_options = data.get("transport", [])
    planner.research_progress = workspace.setdefault("research_progress", {})

    def save_progress():
        # Source events may be observed again during recovery; retain each observation once.
        planner.sources = list({json.dumps(s, sort_keys=True, default=str): s for s in planner.sources}.values())
        data.update(region=planner.geo, sources=planner.sources)
        workspace["issues"] = list(dict.fromkeys(planner.issues))
        if {"days", "places", "transport", "hotel_regions", "budget"} <= set(data):
            from travel_agent.reports.orchestration import validate
            workspace["active_issues"] = validate(planner, data)["issues"]
        if checkpoint:
            checkpoint(workspace)

    planner.checkpoint = save_progress

    def save_notes(notes):
        data["notes"] = list({n["id"]: n for n in [*data.get("notes", []), *notes]}.values())
        save_progress()

    planner.on_notes = save_notes

    def save_points(points):
        data["places"] = list({p["id"]: p for p in [*data.get("places", []), *points]}.values())
        save_progress()

    planner.on_points = save_points
    planner.mandatory_place_ids = set(data.get("mandatory_place_ids", []))
    workspace["result_refs"] = list(dict.fromkeys([*workspace.get("result_refs", []), *args.get("result_refs", [])]))
    planner.provided_results = [
        r for ref in workspace["result_refs"] if (r := database.runtime.result(trip.thread_id, ref))
    ]
    seen = {r["result"].get("result_id") or r.get("execution_id") for r in planner.provided_results}
    for item in database.runtime.index(trip.thread_id, limit=300):
        if item["result_id"] not in seen and item["tool"] in {
            "booking__booking_search", "google_flights__search_flights", "xiaohongshu__get_feed_detail", "web_search"
        } and item["success"]:
            record = database.runtime.result(trip.thread_id, item["result_id"])
            if record:
                planner.provided_results.append(record)
                seen.add(item["result_id"])
    recovered_notes = []
    for record in planner.provided_results:
        if record.get("tool_name") == "web_search":
            for page in record["result"]["data"].get("results", []):
                content = page.get("body") or page.get("content") or page.get("text") or page.get("description") or page.get("snippet")
                url = page.get("url")
                if content and url:
                    recovered_notes.append({"id": "web:" + hashlib.sha256(url.encode()).hexdigest()[:16],
                                            "title": page.get("title", "网页检索摘录"), "content": str(content), "url": url,
                                            "source": "网页检索摘录（非完整笔记正文）"})
            continue
        if record.get("tool_name") != "xiaohongshu__get_feed_detail":
            continue
        value = record["result"]["data"]
        note = value.get("data", value).get("note", {})
        feed_id = record.get("arguments", {}).get("feed_id")
        if feed_id and note.get("desc"):
            recovered_notes.append({"id": feed_id, "title": note.get("title", ""),
                "content": note["desc"], "url": f"https://www.xiaohongshu.com/explore/{feed_id}", "source": "小红书正文"})
    if recovered_notes:
        data["notes"] = list({n["id"]: n for n in [*recovered_notes, *data.get("notes", [])]}.values())
    runtime = database.runtime.load(trip.thread_id)
    planner.selected_results = runtime.get("selected_results", [])
    for selected in planner.selected_results:
        if selected["result_id"] not in workspace["result_refs"]:
            r = database.runtime.result(trip.thread_id, selected["result_id"])
            if r:
                planner.provided_results.append(r)
    targets = args.get("targets") or workspace.get("targets", [])
    planner.hotel_targets = targets if operation == "hotels" else []
    planner.change_request = {
        "target_transport_ids": workspace.get("target_transport_ids", []),
        "transport_target": workspace.get("transport_target", "all"),
    }
    if operation != "route" and not planner.geo:
        return {"success": False, "next": "缺少城市与日期骨架，先route"}, workspace
    if operation in {"schedule", "hotels", "render"}:
        transport = data.get("transport", [])
        cities = planner.geo.get("cities", [])
        missing = []
        if planner.geo.get("entry_mode") not in {None, "none"} and trip.origin and not any(
            t.get("scope") == "external" and t.get("options") for t in transport
        ):
            missing.append("出发地往返交通尚未关联报告")
        pairs = {(t.get("origin_id"), t.get("destination_id")) for t in transport
                 if t.get("scope") == "intercity" and t.get("options")}
        missing += [f"{a['name']} → {b['name']}城际交通" for a, b in zip(cities, cities[1:])
                    if (a["id"], b["id"]) not in pairs]
        if missing:
            return {"success": False, "error_type": "missing_dependency", "missing": missing,
                    "next": "先用transport关联已查航班并补齐交通段；日期和可用时间明确后再排程/查住宿。复用已有result_refs，不重新搜索已选航班。"}, workspace
        if operation in {"hotels", "render"}:
            signature = schedule_signature(planner.geo, data, workspace)
            if workspace.get("schedule_signature") == schedule_signature(planner.geo, data, workspace, legacy=True):
                workspace["schedule_signature"] = signature
            if workspace.get("schedule_signature") and workspace["schedule_signature"] != signature:
                return {"success": False, "error_type": "stale_schedule",
                        "next": "交通或点位已变，先schedule更新实际日期和可用时间，再查住宿；保留未受影响证据。"}, workspace
    if operation == "route":
        try:
            planner.geo = resolve_route(planner, old)
        except ApprovedRouteConflict as exc:
            return {"success": False, "error_type": "approved_route_conflict",
                    "error": str(exc), "issues": exc.issues}, workspace
        if drift:
            # Results are still indexed by ID, but old route-derived products
            # cannot certify transport, stays or scheduling for a corrected route.
            for k in ("transport", "days", "hotel_regions", "budget", "food"):
                data.pop(k, None)
            for k in ("transport_progress", "flight_alignment_signature", "schedule_signature", "active_issues"):
                workspace.pop(k, None)
            workspace["completed"] = [k for k in workspace["completed"]
                                      if k not in {"route", "research", "transport", "schedule", "hotels"}]
            workspace["sections"] = sorted(scope | {"route", "days", "hotel_regions", "transport", "budget"})
        data["region"] = planner.geo
        allowed = {c["id"] for c in planner.geo["cities"]}
        data["places"] = [p for p in data.get("places", []) if p.get("city_id") in allowed]
    elif operation == "research":
        notes, points, insights = data.get("notes", []), data.get("places", []), data.get("insights", [])
        cities = [
            c for c in planner.geo["cities"]
            if not targets or c["name"] in targets or c["id"] in targets
        ]
        if not cities:
            return {"success": False, "next": "targets未对应现有城市，请使用城市名称或ID"}, workspace
        purpose = args.get("purpose", "all")
        pending = [c for c in cities if not planner.research_progress.get(c["id"], {}).get("completed")
                   or purpose in {"food", "repair", "coverage_repair"}]
        for city in pending[:1]:
            existing = [p for p in points if p.get("city_id") == city["id"]]
            notes, new, extra = research_city(
                planner, city, notes, existing, purpose=purpose
            )
            points = [p for p in points if p.get("city_id") != city["id"]] + new
            insights += extra
            planner.research_progress[city["id"]].update(completed=True, phase="completed", places=len(new))
            data.update(notes=notes, places=points, insights=insights)
            save_progress()
        data.update(notes=notes, places=points, insights=insights)
        data["selected_place_ids"] = list(dict.fromkeys([
            *data.get("selected_place_ids", []), *(p["id"] for p in points),
        ]))
        remaining = [c["name"] for c in cities if not planner.research_progress.get(c["id"], {}).get("completed")]
        if remaining:
            save_progress()
            return {"success": True, "operation": operation, "unit_completed": city["name"],
                    "pending_targets": remaining, "places": len(points), "notes": len(notes),
                    "next": "成果已保存，继续research pending_targets，不能视为整份研究已完成"}, workspace
    elif operation == "transport":
        from travel_agent.reports.mobility import refresh_transfers
        transport = data.get("transport", [])
        previous_windows = external_windows(transport)
        target = args.get("transport_target") or workspace.get("transport_target", "all")
        if target == "intercity" and len(planner.geo["cities"]) < 2 and not transport:
            return {
                "success": False, "error_type": "wrong_transport_scope",
                "next": "只有一个目的地，没有旅途内部城际段；出发地往返该地属于external，请查询external",
            }, workspace
        external = [t for t in transport if t.get("scope") == "external"]
        first_date, last_date = trip.travel_dates.start_date, trip.travel_dates.end_date
        if first_date and last_date and any(
            not first_date.isoformat() <= str(t.get("date") or "") <= last_date.isoformat() or not t.get("options")
            for t in external
        ):
            external = []
        internal = [t for t in transport if t.get("scope") == "intercity"]
        target_ids = args.get("target_transport_ids") or workspace.get("target_transport_ids", [])
        if target_ids:
            matching = [t for t in transport if t.get("id") in target_ids]
            if len(matching) != len(target_ids):
                return {"success": False, "next": "目标交通段ID不存在，请先读取现有交通"}, workspace
            kinds = {t.get("scope") for t in matching}
            target = next(iter(kinds)) if len(kinds) == 1 else "all"
        if target != "intercity" and (not external or target_ids):
            if len({c["country_code"] for c in planner.geo["cities"]}) > 1:
                from travel_agent.reports.mobility import plan_entry_exit
                external = plan_entry_exit(planner, planner.geo["cities"])
            else:
                external = external_transport(planner, route_days(planner))
            data["transport"] = external + internal
            save_progress()
        if not old or "route" in scope:
            windows = [{k: v.get(k) for k in ("departure_time", "arrival_time")} for v in external_windows(external)]
            signature = canonical([windows, trip.hard_constraints, [[c["id"], c["days"]] for c in planner.geo["cities"]]])
            if workspace.get("flight_alignment_signature") != signature:
                try:
                    fit_route_to_flights(planner, external)
                except ApprovedRouteConflict as exc:
                    return {"success": False, "error_type": "approved_route_conflict",
                            "error": str(exc), "issues": exc.issues}, workspace
                workspace["flight_alignment_signature"] = canonical([
                    windows, trip.hard_constraints, [[c["id"], c["days"]] for c in planner.geo["cities"]],
                ])
            data["region"] = planner.geo
        if target != "external":
            stays = copy.deepcopy(planner.geo["cities"])
            # A round trip returns through its actual departure city; open-jaw legs
            # are researched by plan_entry_exit for multi-country routes.
            if (
                planner.geo.get("entry_mode") == "flight"
                and stays[-1]["id"] != stays[0]["id"]
                and len({c["country_code"] for c in stays}) == 1
            ):
                _, returning = external_windows(external)
                closing = copy.deepcopy(stays[0])
                cutoff = (
                    datetime.fromisoformat(returning["departure_time"]) - timedelta(hours=3)
                    if returning.get("departure_time") else datetime.combine(trip.travel_dates.end_date, datetime.min.time())
                )
                closing.update(checkin=cutoff.date().isoformat(), start_day=(cutoff.date() - trip.travel_dates.start_date).days + 1)
                stays.append(closing)
            if target_ids and all(i in {e.get("id") for e in internal} for i in target_ids):
                internal = refresh_transfers(planner, internal, target_ids)
            else:
                from travel_agent.reports.mobility import _edges, _Research
                edges = _edges(planner, stays, False)
                progress = workspace.setdefault("transport_progress", [])
                have = {t["id"] for t in internal if t.get("options")} | set(progress)
                pending_edges = [e for e in edges if e["id"] not in have]
                for edge in pending_edges[:1]:
                    updated = _Research(planner, [edge]).run()
                    replaced = {e["id"] for e in updated}
                    internal = [e for e in internal if e.get("id") not in replaced] + updated
                    if edge["id"] not in progress:
                        progress.append(edge["id"])
                    data["transport"] = external + internal
                    save_progress()
                if len(pending_edges) > 1:
                    return {"success": True, "operation": operation, "pending_segments": [
                        {k: e.get(k) for k in ("id", "origin", "destination", "date")} for e in pending_edges[1:]
                    ], "next": "当前交通段已保存，继续transport完成剩余城际段"}, workspace
        data["transport"] = external + internal
        if old and previous_windows != external_windows(data["transport"]):
            scope.add("days")
            workspace["sections"] = sorted(scope)
    elif operation == "schedule":
        from travel_agent.reports.adjustments import align_entry_transfer, apply_adjustments
        align_entry_transfer(planner, data, workspace)
        apply_adjustments(planner, data, workspace, args)
        planner.transport_options = data.get("transport", [])
        bases = {c["id"]: c for c in planner.geo["cities"]}
        original_points = data.get("places", [])
        points = [p for p in original_points if matches_base_area(bases.get(p.get("city_id")), p)]
        if len(points) != len(original_points):
            data["places"] = points
            known_ids = {p["id"] for p in points}
            if "selected_place_ids" in data:
                data["selected_place_ids"] = [identifier for identifier in data["selected_place_ids"] if identifier in known_ids]
        if not points:
            return {"success": False, "next": "尚无有来源的体验，先research对应城市"}, workspace
        bind_confirmed_activities(planner, data, workspace)
        data["mandatory_place_ids"] = sorted(planner.mandatory_place_ids)
        planner.friend_time_cities = friend_time_cities(planner)
        data["friend_time_cities"] = sorted(planner.friend_time_cities)
        if "selected_place_ids" in args:
            known = {p["id"]: p for p in points}
            if not args["selected_place_ids"] or set(args["selected_place_ids"]) - set(known):
                return {"success": False, "next": "选择的体验必须先有来源并完成点位核验，请research后使用返回的准确ID"}, workspace
            changed_cities = {known[p].get("city_id") for p in args["selected_place_ids"]}
            previous = data.get("selected_place_ids", list(known))
            # Old runtime versions replaced all cities when given a local selection.
            # Recover unaffected IDs from the last valid report if that damaged a draft.
            for row in database.connection.execute("SELECT payload_json FROM reports WHERE trip_id=? ORDER BY version DESC", (trip.trip_id,)):
                saved = json.loads(row[0]).get("report", {})
                if saved.get("quality", {}).get("passed"):
                    previous = list(dict.fromkeys([*previous, *saved.get("selected_place_ids", [])]))
                    break
            data["selected_place_ids"] = list(dict.fromkeys([
                *(p for p in previous if p in known and known[p].get("city_id") not in changed_cities),
                *args["selected_place_ids"],
            ]))
        selected = set(data.get("selected_place_ids", [p["id"] for p in points])) | planner.mandatory_place_ids
        signature = schedule_signature(planner.geo, data, workspace)
        if workspace.get("schedule_signature") == signature and len(data.get("days", [])) == trip.duration_days:
            gaps = list(dict.fromkeys(d["city"] for d in sparse_tour_days(data["days"], data.get("transport", []))))
            return {"success": not gaps, "unchanged": True, "research_targets": gaps,
                    "next": "排程没有变化。用research的coverage_repair补齐列出的城市，或用确有依据的调整重排。"
                    if gaps else "现有排程可复用，继续检查住宿和报告。"}, workspace
        previous_signatures = {signature, schedule_signature(planner.geo, data, workspace, legacy=True)}
        planner.previous_scheduled_days = copy.deepcopy(data.get("days", [])) if workspace.get("schedule_signature") in previous_signatures else []
        workspace["schedule_signature"] = signature

        def save_days(days):
            data["days"] = copy.deepcopy(days)
            save_progress()

        planner.on_days = save_days
        planner.stay_overrides = workspace.get("stay_overrides", {})
        data["days"] = build_days(planner, [p for p in points if p["id"] in selected], data.get("transport", []))
        for day in data["days"]:
            if str(day["day"]) in workspace.get("stay_overrides", {}) and not day.get("no_hotel"):
                day["stay_city"] = workspace["stay_overrides"][str(day["day"])]
        valid_stays = {
            (g["name"], g["checkin"], g["checkout"])
            for g in stays_for_days(planner, data["days"], data.get("transport", []))
        }
        for hotel in data.get("hotel_regions", []):
            if (hotel["city"], hotel.get("checkin"), hotel.get("checkout")) not in valid_stays:
                hotel["compatible"] = False
                hotel["refresh_status"] = "交通或日程改变了住宿日期，原候选仅作历史参考。"
    elif operation == "hotels":
        if not data.get("days"):
            return {"success": False, "next": "先schedule确定实际住宿日期"}, workspace
        # hotels() uses planner.previous and hotel_targets to preserve unrelated stays.
        planner.hotel_choices = {x["city"]: x["url"] for x in args.get("hotel_choices", [])}
        planner.previous = {**old, "hotel_regions": data.get("hotel_regions", old.get("hotel_regions", []))}

        def save_hotels(groups):
            data["hotel_regions"] = copy.deepcopy(groups)
            save_progress()

        planner.on_hotels = save_hotels
        stays = stays_for_days(planner, data["days"], data.get("transport", []))
        pending_stays = [g for g in stays if (not targets or g["name"] in targets)
                         and not any(h.get("city") == g["name"] and h.get("checkin") == g["checkin"]
                                     and h.get("checkout") == g["checkout"] and h.get("hotels")
                                     for h in data.get("hotel_regions", []))]
        if pending_stays:
            planner.hotel_targets = [pending_stays[0]["name"]]
        data["hotel_regions"] = planner.hotels(data["days"])
        save_progress()
        if len(pending_stays) > 1:
            return {"success": True, "operation": operation, "pending_targets": [g["name"] for g in pending_stays[1:]],
                    "next": "本住宿段已保存，继续hotels处理剩余片区"}, workspace
    elif operation == "render":
        required = {"days", "places", "transport", "hotel_regions"}
        missing = [k for k in required if k not in data]
        if planner.geo.get("entry_mode") != "none" and trip.origin and not data.get("transport"):
            return {"success": False, "next": "尚无出发地往返交通，请先transport，transport_target=external或all"}, workspace
        needed = set()
        done = set(workspace["completed"])
        if "route" in scope:
            needed |= {"route", "transport", "schedule", "hotels"}
            if any(
                not any(p.get("city_id") == city["id"] for p in data.get("places", []))
                for city in planner.geo["cities"]
            ):
                needed.add("research")
        else:
            for section, step in {
                "days": "schedule", "food": "research", "notes": "research",
                "hotel_regions": "hotels", "transport": "transport",
            }.items():
                if section in scope:
                    needed.add(step)
        if missing or needed - done:
            return {"success": False, "missing_data": missing, "pending_operations": sorted(needed - done)}, workspace
        data.update(
            region=planner.geo, sources=planner.sources,
            food=[p for p in data["places"] if p.get("kind") in {"meal", "cafe"}],
            trip=trip.model_dump(mode="json"),
            point_count=sum(len(d.get("stops", [])) for d in data["days"]),
        )
        data["budget"] = build_budget(planner, data)
        plan, _ = finish(planner, data, scope, "按当前任务组合报告能力", old)
        trip.report_version = max(trip.report_version, planner.base_version) + 1
        plan.report["trip"] = trip.model_dump(mode="json")
        database.save_trip(trip)
        markdown, html = renderer.render(plan)
        database.append_event(trip.thread_id, trip.user_id, trip.trip_id, "report.generated", {
            "markdown_path": str(markdown), "html_path": str(html),
            "quality": plan.report["quality"], "revision": plan.report["revision"],
        })
        workspace["rendered"] = True
        output = {
            "success": True, "html_path": str(html), "markdown_path": str(markdown),
            "quality": plan.report["quality"], "budget": plan.report["budget"],
            "revision": plan.report["revision"],
            "days": [{"day": d["day"], "city": d["city"],
                      "places": [p["name"] for p in d["stops"]]} for d in data["days"]],
        }
        workspace["artifacts"] = output
        return output, workspace
    else:
        return {"success": False, "error": "未知报告操作"}, workspace
    data.update(region=planner.geo, sources=planner.sources)
    workspace["issues"] = planner.issues
    workspace["completed"] = list(dict.fromkeys([*workspace["completed"], operation]))
    save_progress()
    return {
        "success": True, "operation": operation, "completed": workspace["completed"],
        "cities": [{**{k: c.get(k) for k in ("id", "name", "days", "country_code")},
                    "visit_days": c["days"] - len(reserve_kinds(c))}
                   for c in planner.geo["cities"]],
        "places": len(data.get("places", [])), "notes": len(data.get("notes", [])),
        "days": len(data.get("days", [])),
        "hotels": len(data.get("hotel_regions", [])),
        "transport": [{k: t.get(k) for k in ("id", "origin", "destination", "scope")} for t in data.get("transport", [])],
        "issues": workspace.get("active_issues", []),
        "research_notes": planner.issues[-3:],
        "timing_issues": [{"day": day["day"], "issue": day["timing_warning"]} for day in data.get("days", []) if day.get("timing_warning")],
        "available_experiences": [{k: p.get(k) for k in ("id", "name", "kind", "city", "duration_minutes")}
                                  for p in data.get("places", []) if operation == "schedule" and (not targets or p.get("city") in targets)],
        "schedule_method": "schedule仅安排已有点位，request文字不会创建新活动。新增体验先research，purpose=coverage_repair定向补查，并通过result_refs复用新检索证据；再schedule，必要时用selected_place_ids选择真实点位。" if operation == "schedule" else None,
        "next": "若要调整已存道路接送时刻，用schedule的transfer_changes传准确ID、date、departure_time；只写request不会改时刻。改过夜城市用stay_changes，然后按受影响日期hotels。已有交通完整时无需再次transport检索。",
    }, workspace
