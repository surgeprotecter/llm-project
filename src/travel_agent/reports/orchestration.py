"""Adaptive report loop: identify objects, research gaps, verify, revise snapshots."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

from travel_agent.models import ItineraryDay, TravelPlan
from travel_agent.reports.costs import build_budget
from travel_agent.reports.daily import build_days
from travel_agent.reports.research import research_city
from travel_agent.reports.revision import SECTIONS, preserve_failed_refresh, revision_scope
from travel_agent.reports.route import (
    canonical_city,
    chosen_option,
    external_windows,
    preserve_route_notes,
    resolve_route,
    route_days,
    stays_for_days,
)


def external_transport(planner, skeleton):
    mode = planner.geo.get("entry_mode", "flight")
    if mode == "none" or not planner.trip.origin:
        return []
    if mode == "rail":
        first, last = planner.geo["cities"][0]["name"], planner.geo["cities"][-1]["name"]
        result = [planner.rail(planner.trip.origin, first, skeleton[0]["date"])]
        if last != planner.trip.origin:
            result.append(planner.rail(last, planner.trip.origin, skeleton[-1]["date"]))
    elif mode == "flight":
        # The route, not a keyword in a user message, decides transport capability.
        old_query = planner.query
        try:
            planner.query = "查询适合当前旅行要求的往返航班。" + (
                "舒适、少转机。"
                if planner.trip.preferences.get("flight")
                or planner.trip.preferences.get("flight_preferences")
                else ""
            )
            result = planner.transport(skeleton)
        finally:
            planner.query = old_query
    else:
        return [
            {
                "id": "external:road",
                "scope": "external",
                "type": "往返地面交通",
                "origin": planner.trip.origin,
                "destination": planner.geo["cities"][0]["name"],
                "date": skeleton[0]["date"],
                "options": [],
                "url": "",
                "status": "unavailable",
                "raw_summary": "需补查出入城市的地面交通方式与费用。",
            }
        ]
    for index, item in enumerate(result):
        item.update(
            id=f"external:{index}",
            kind="entry" if index == 0 else "exit",
            mode="rail" if mode == "rail" else "flight" if mode == "flight" else "road",
            scope="external",
            status="quote" if item.get("options") else "unavailable",
        )
    return result


def validate(planner, report):
    from travel_agent.reports.delivery import completeness_issues

    issues = [
        *getattr(planner, "unfulfilled_requests", []),
        *(f"已确认体验尚缺可核验候选：{item}" for item in report.get("unfulfilled_activities", [])),
        *completeness_issues(report),
    ]
    if planner.geo.get("entry_mode") != "none" and planner.trip.origin and not any(
        t.get("scope") == "external" for t in report["transport"]
    ):
        issues.append("缺少出发地与目的地之间的往返交通，不能视为完整行程。")
    expected_stays = stays_for_days(planner, report["days"], report["transport"])
    if expected_stays and not report["hotel_regions"]:
        issues.append("有实际住宿夜晚但未提供住宿候选。")
    for day in report["days"]:
        if day.get("timing_warning"):
            issues.append(f"第{day['day']}天：{day['timing_warning']}")
        if day.get("research_gap"):
            issues.append(f"第{day['day']}天（{day['city']}）：缺少可定位的具体游览或活动。")
        if day.get("transfer_unresolved"):
            issues.append(f"第{day['day']}天：城际交通时长未落实，日程只作草案。")
        if any(not leg.get("verified") for leg in day["legs"]):
            issues.append(f"第{day['day']}天：部分地点间道路耗时未核验。")
        blocks = [
            b for b in day.get("timeline", []) if b.get("start_time") and b.get("duration_minutes")
        ]
        blocks.sort(key=lambda b: b["start_time"])
        for a, b in zip(blocks, blocks[1:]):
            ah, am = map(int, a["start_time"].split(":"))
            bh, bm = map(int, b["start_time"].split(":"))
            if ah * 60 + am + int(a["duration_minutes"]) > bh * 60 + bm:
                issues.append(f"第{day['day']}天：{a['name']}与{b['name']}时间重叠。")
    from travel_agent.reports.daily import sparse_tour_days

    issues += [
        f"第{d['day']}天（{d['city']}）：整日只有短暂停留，缺少有来源的半日体验或明确的休闲安排。"
        for d in sparse_tour_days(report["days"], report["transport"])
    ]
    for stay in report["hotel_regions"]:
        if stay.get("compatible") is False:
            issues.append(f"{stay['city']}住宿日期已与当前交通不兼容，需补查。")
        if not stay.get("hotels"):
            issues.append(
                f"{stay['city']} {stay.get('checkin', '')}至{stay.get('checkout', '')}缺少可用住宿候选。"
            )
        elif not any(h.get("price_is_total") for h in stay["hotels"]):
            issues.append(f"{stay['city']}住宿未取得计价范围明确的报价，预算暂留。")
    for edge in report["transport"]:
        if not edge.get("options"):
            issues.append(
                f"{edge['origin']} → {edge['destination']}交通尚未落实：{edge.get('raw_summary', '需补查')}。"
            )
        elif edge.get("scope") == "intercity" and not (
            chosen_option(edge).get("duration_minutes")
            or chosen_option(edge).get("planning_duration_minutes")
        ):
            issues.append(f"{edge['origin']} → {edge['destination']}缺少有依据的通行时间。")
    scheduled_ids = {s.get("id") for d in report["days"] for s in d["stops"]}
    for identifier in report.get("mandatory_place_ids", []):
        if identifier not in scheduled_ids:
            point = next((p for p in report["places"] if p["id"] == identifier), {})
            issues.append(
                f"指定体验未能排入当前可用时间：{point.get('name', identifier)}；需要调整停留时长或取舍。"
            )
    city_ids = {d.get("city_id") for d in report["days"] if not d.get("type") == "transfer"}
    for city in report["region"]["cities"]:
        if city["id"] in city_ids and not any(
            p.get("city_id") == city["id"] for p in report["places"]
        ):
            issues.append(f"{city['name']}研究未覆盖，不能视为完整行程。")
    _, homebound = external_windows(report["transport"])
    if homebound.get("arrival_time", "")[:10] > planner.trip.travel_dates.end_date.isoformat():
        issues.append(
            f"所选返程于{homebound['arrival_time']}抵达出发地，晚于当前旅行结束日期，需确认日期口径或换更早航班。"
        )
    # These checks concern only route allocation. Old model comments about later,
    # not-yet-populated transport/hotel JSON are not final quality observations.
    checks = report["region"].get("constraint_checks", [])
    issues += [
        f"用户路线约束未满足：{c['requirement']}。{c.get('reason', '')}"
        for c in checks
        if c.get("scope") == "route" and c.get("satisfied") is False
    ]
    ceiling = planner.trip.budget.max_amount
    if ceiling is not None and report.get("budget", {}).get("low", 0) > ceiling:
        issues.append("当前估算下限已经超过用户明确的预算上限，需要降低酒店/交通等费用或调整行程。")
    return {"passed": not issues, "issues": list(dict.fromkeys(issues))}


def route_takeaways(report):
    """Describe the current material, never concatenate superseded planning prose."""
    statements = []
    stays = report.get("hotel_regions", [])
    if stays:
        statements.append("实际住宿顺序：" + " → ".join(
            f"{g['city']}（{g.get('nights', 0)}晚）" for g in stays
        ) + "。")
    incoming, returning = external_windows(report.get("transport", []))
    if incoming.get("arrival_time"):
        statements.append(f"抵达目的地：{incoming['arrival_time'].replace('T', ' ')}，以所选航班当地时间为准。")
    if returning.get("departure_time"):
        statements.append(f"返程起飞：{returning['departure_time'].replace('T', ' ')}，以所选航班当地时间为准。")
    estimated = [t for t in report.get("transport", []) if t.get("scope") == "intercity"
                 and chosen_option(t).get("status") == "estimate"]
    if estimated:
        statements.append(f"{len(estimated)}段城际交通使用估算，具体方式、费用与可订状态请逐段复核。")
    return statements


def finish(planner, report, scope, reason, old):
    preserve_route_notes(planner.geo)
    report.update(
        schema_version=2,
        research_version=3,
        generated_at=datetime.now(UTC).isoformat(),
        trip=planner.trip.model_dump(mode="json"),
        region=planner.geo,
        sources=planner.sources,
        point_count=sum(len(d["stops"]) for d in report["days"]),
        travel_mode=planner.geo.get("entry_mode", "flight"),
    )
    report["quality"] = validate(planner, report)
    report["status"] = (
        "真实检索 · 可执行候选方案" if report["quality"]["passed"] else "草案 · 尚有信息缺口"
    )
    report["summary"] = (
        f"{len(planner.geo['cities'])}个城市／区域，按实际抵离时间组织{planner.trip.duration_days}天，吃喝与体验融入每日安排。"
    )
    report["route_takeaways"] = route_takeaways(report)
    report["research_notes"] = list(dict.fromkeys(planner.issues))
    report["warnings"] = list(report["quality"]["issues"])
    report["revision"] = {
        "base_version": planner.base_version,
        "requested_sections": sorted(scope),
        "changed_sections": sorted(k for k in SECTIONS if report.get(k) != old.get(k)),
        "preserved_sections": sorted(
            k for k in SECTIONS if k in old and report.get(k) == old.get(k)
        ),
        "reason": reason,
        "query": planner.query,
        "tasks": getattr(planner, "change_request", {}).get("tasks", []),
        "target_transport_ids": getattr(planner, "change_request", {}).get(
            "target_transport_ids", []
        ),
        "transport_target": getattr(planner, "change_request", {}).get("transport_target"),
    }
    report["confirmations"] = [
        "报价未锁定，出发前复核交通库存、行李及酒店房型税费。",
        "根据活动体力需求、天气和营业时间复核每日时间安排。",
    ]
    planner.db.append_event(
        planner.thread_id,
        planner.trip.user_id,
        planner.trip.trip_id,
        "report.revision_applied",
        report["revision"],
    )
    planner.trip.selected_places = report["places"]
    planner.trip.route_plan = report["days"]
    planner.trip.hotel_options = [h for g in report["hotel_regions"] for h in g["hotels"]]
    planner.trip.transport_options = report["transport"]
    planner.db.save_trip(planner.trip)
    return TravelPlan(
        trip_id=planner.trip.trip_id,
        title=f"{planner.trip.destination} {planner.trip.duration_days}日旅行计划",
        destination=planner.trip.destination,
        summary=report["summary"],
        itinerary=[
            ItineraryDay(day=d["day"], title=d["title"], activities=d.get("timeline", d["stops"]))
            for d in report["days"]
        ],
        hotel_options=planner.trip.hotel_options,
        transport_options=report["transport"],
        research_notes=report["notes"],
        evidence_ids=planner.trip.evidence_ids,
        sources=planner.sources,
        warnings=report["warnings"],
        report=report,
    ), planner.results


def run_adaptive(planner):
    from travel_agent.reports.route import upgrade_snapshot

    old = upgrade_snapshot(planner.previous)
    scope, reason = revision_scope(planner, old)
    content_requested = bool(scope & {"route", "days", "notes", "food"})
    if old.get("trip", {}).get("destinations") != planner.trip.destinations:
        old = {}
    planner.sources = copy.deepcopy(old.get("sources", []))
    planner.geo = (
        resolve_route(planner, old) if "route" in scope or not old else copy.deepcopy(old["region"])
    )
    notes = copy.deepcopy(old.get("notes", []))
    points = copy.deepcopy(old.get("places", []))
    insights = copy.deepcopy(old.get("insights", []))
    if scope & {"route", "days", "notes", "food"}:
        selected = getattr(planner, "change_request", {}).get("research_cities", [])
        for city in planner.geo["cities"]:
            existing = [p for p in points if p.get("city_id") == city["id"]]
            primary_count = sum(
                p.get("kind", "sightseeing") in {"sightseeing", "activity", "shop"}
                for p in existing
            )
            needed = bool(
                not existing
                or scope & {"notes", "food"}
                or city["name"] in selected
                or ("route" in scope and primary_count < max(2, int(city["days"]) + 1))
            )
            if selected and city["name"] not in selected and existing:
                needed = False
            if needed:
                notes, new, new_insights = research_city(
                    planner,
                    city,
                    notes,
                    existing,
                    purpose="food" if scope <= {"food", "budget"} else "all",
                )
                points = [p for p in points if p.get("city_id") != city["id"]] + new
                insights += new_insights
    allowed = {c["id"] for c in planner.geo["cities"]}
    points = [p for p in points if p.get("city_id") in allowed]
    selected_ids = old.get("selected_place_ids", [p["id"] for p in points])
    mandatory_ids = old.get("mandatory_place_ids", [])
    if old and scope & {"route", "days", "notes"}:
        selection = planner.ask(
            "根据用户当前修改与硬约束，从已核验体验选择允许进入日程的ID，按推荐先后排列。"
            "用户只去指定景点时必须排除该城市其他景点；未要求修改城市的体验保留。"
            "不能造ID，也不能用已排除景点凑数；餐饮和商店仍按偏好保留。"
            '返回 {"allowed_ids":[string],"mandatory_ids":[string],"missing_places":[{"name":用户明确提到的名称,"city":string,"query_name":英文检索名}],"unfulfilled_requests":[string]}。',
            {
                "query": planner.query,
                "constraints": planner.trip.hard_constraints,
                "places": [{k: p.get(k) for k in ("id", "name", "city", "kind")} for p in points],
            },
            "experience_selection",
        )
        selected_ids = [
            i for i in selection.get("allowed_ids", []) if any(p["id"] == i for p in points)
        ]
        mandatory_ids = [
            i for i in selection.get("mandatory_ids", []) if any(p["id"] == i for p in points)
        ]
        if selection.get("missing_places"):
            from travel_agent.reports.research import research_requested

            notes, recovered, gaps = research_requested(planner, selection["missing_places"], notes)
            points = list({p["id"]: p for p in [*points, *recovered]}.values())
            selected_ids = list(dict.fromkeys([*selected_ids, *(p["id"] for p in recovered)]))
            mandatory_ids = list(dict.fromkeys([*mandatory_ids, *(p["id"] for p in recovered)]))
            planner.unfulfilled_requests = gaps
        planner.issues.extend(selection.get("unfulfilled_requests", []))
    elif not old.get("selected_place_ids"):
        selected_ids = [p["id"] for p in points]
    elif "food" in scope:
        selected_ids = list(
            dict.fromkeys(
                [*selected_ids, *(p["id"] for p in points if p.get("kind") in {"meal", "cafe"})]
            )
        )
    # A transport or hotel edit must never reintroduce excluded sightseeing IDs.
    planner.mandatory_place_ids = set(mandatory_ids)
    scheduled_points = sorted(
        [p for p in points if p["id"] in selected_ids], key=lambda p: selected_ids.index(p["id"])
    )
    transport = copy.deepcopy(old.get("transport", []))
    target = getattr(planner, "change_request", {}).get("transport_target", "all")
    skeleton = route_days(planner)
    if "transport" in scope:
        external = [t for t in transport if t.get("scope") == "external"]
        if target != "intercity" or "route" in scope or not old:
            fresh = external_transport(planner, skeleton)
            external = preserve_failed_refresh(
                fresh, external, "出入境/往返交通", planner.issues, compatible="route" not in scope
            )
        previous_flight_windows = external_windows(old.get("transport", []))
        if (
            "route" in scope or not old or previous_flight_windows != external_windows(external)
        ) and target != "external":
            from travel_agent.reports.route import fit_route_to_flights

            if fit_route_to_flights(planner, external):
                scope.update(("route", "days", "hotel_regions", "budget"))
        internal = [t for t in transport if t.get("scope") == "intercity"]
        if target != "external" or "route" in scope or not old:
            from travel_agent.reports.mobility import plan_intercity

            transfer_stays = copy.deepcopy(planner.geo["cities"])
            if (
                planner.geo.get("entry_mode") == "flight"
                and transfer_stays[-1]["id"] != transfer_stays[0]["id"]
            ):
                closing = copy.deepcopy(transfer_stays[0])
                _, departing = external_windows(external)
                departure_limit = (
                    (datetime.fromisoformat(departing["departure_time"]) - timedelta(hours=3))
                    if departing.get("departure_time")
                    else None
                )
                closing["checkin"] = (
                    departure_limit.date().isoformat()
                    if departure_limit
                    else planner.trip.travel_dates.end_date.isoformat()
                )
                closing["start_day"] = (
                    datetime.fromisoformat(closing["checkin"]).date()
                    - planner.trip.travel_dates.start_date
                ).days + 1
                transfer_stays.append(closing)
            target_ids = getattr(planner, "change_request", {}).get("target_transport_ids", [])
            known = {e.get("id") for e in internal}
            if target_ids and "route" not in scope and set(target_ids) <= known:
                from travel_agent.reports.mobility import refresh_transfers

                internal = refresh_transfers(planner, internal, target_ids)
            else:
                internal = plan_intercity(planner, transfer_stays)
        transport = external + internal
    planner.transport_options = transport
    # A flight-only request keeps existing researched places/hotels; only retime days if windows changed.
    old_windows = external_windows(old.get("transport", []))
    new_windows = external_windows(transport)
    windows_changed = [
        {k: d.get(k) for k in ("departure_time", "arrival_time")} for d in old_windows
    ] != [{k: d.get(k) for k in ("departure_time", "arrival_time")} for d in new_windows]

    def transfer_signature(values):
        return [
            (
                t.get("id"),
                t.get("date"),
                {
                    k: chosen_option(t).get(k)
                    for k in (
                        "departure_date",
                        "arrival_date",
                        "departure_time",
                        "arrival_time",
                        "duration_minutes",
                    )
                },
            )
            for t in values
            if t.get("scope") == "intercity"
        ]

    internal_changed = transfer_signature(transport) != transfer_signature(old.get("transport", []))
    days = (
        build_days(planner, scheduled_points, transport)
        if scope & {"route", "days", "food"} or not old or windows_changed or internal_changed
        else copy.deepcopy(old["days"])
    )
    from travel_agent.reports.daily import sparse_tour_days

    if content_requested:
        repair_ids = {d["city_id"] for d in sparse_tour_days(days, transport)}
        for city in planner.geo["cities"]:
            if city["id"] in repair_ids:
                existing = [p for p in points if p.get("city_id") == city["id"]]
                notes, enriched, new_insights = research_city(
                    planner, city, notes, existing, purpose="coverage_repair"
                )
                points = list({p["id"]: p for p in [*points, *enriched]}.values())
                insights += new_insights
                selected_ids = list(dict.fromkeys([*selected_ids, *(p["id"] for p in enriched)]))
        if repair_ids:
            scheduled_points = sorted(
                [p for p in points if p["id"] in selected_ids],
                key=lambda p: selected_ids.index(p["id"]),
            )
            days = build_days(planner, scheduled_points, transport)
    if (windows_changed or internal_changed) and old:
        scope.add("days")
    _, departing = external_windows(transport)
    if departing.get("departure_time"):
        for day in days:
            if day.get("type") == "transfer" and day["date"] == departing["departure_time"][:10]:
                city = planner.geo["cities"][0]
                day.update(city=city["name"], city_id=city["id"], area=city["name"], stay_city=None)
    stays = stays_for_days(planner, days, transport)
    old_stays = [
        (g["city"], g.get("checkin"), g.get("checkout")) for g in old.get("hotel_regions", [])
    ]
    new_stays = [(g["name"], g["checkin"], g["checkout"]) for g in stays]
    hotels = copy.deepcopy(old.get("hotel_regions", []))
    # A changed flight arrival does not fetch new hotels unless the actual nights changed.
    if "hotel_regions" in scope or (
        old_stays != new_stays and not (old and scope <= {"transport", "budget", "days"})
    ):
        scope.add("hotel_regions")
        hotels = planner.hotels(days)
        for group in hotels:
            city = canonical_city(planner.geo, group["city"])
            group["id"] = f"{city['id'] if city else group['city']}:{group.get('checkin', '')}"
    if old_stays != new_stays and "hotel_regions" not in scope:
        for group in hotels:
            if (group["city"], group.get("checkin"), group.get("checkout")) not in new_stays:
                group.update(
                    compatible=False,
                    refresh_status="航班改变了实际住宿夜晚，此项仅保留历史参考，需按新日期补查。",
                )
        planner.issues.append(
            "当前航班改变了住宿日期；本轮按用户要求仅查询航班，旧酒店报价不计为兼容报价。"
        )
    food = [p for p in points if p.get("kind") in {"meal", "cafe"}]
    report = {
        **copy.deepcopy(old),
        "days": days,
        "places": points,
        "selected_place_ids": selected_ids,
        "mandatory_place_ids": mandatory_ids,
        "notes": list({n["id"]: n for n in notes}.values()),
        "insights": list({(i["title"], i["text"]): i for i in insights}.values()),
        "hotel_regions": hotels,
        "transport": transport,
        "food": food,
        "trip": planner.trip.model_dump(mode="json"),
        "region": planner.geo,
        "point_count": sum(len(d["stops"]) for d in days),
    }
    report["budget"] = (
        build_budget(planner, report) if "budget" in scope or not old else old["budget"]
    )
    return finish(planner, report, scope, reason, old)
