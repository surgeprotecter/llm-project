"""Apply model-chosen planning times while retaining immutable provider facts."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta

from travel_agent.reports.route import canonical_city, chosen_option, external_windows


def align_entry_transfer(planner, data, workspace):
    """An unassigned road pickup cannot depart before the actual inbound flight."""
    incoming, _ = external_windows(data.get("transport", []))
    if not incoming.get("arrival_time"):
        return
    earliest = datetime.fromisoformat(incoming["arrival_time"]) + timedelta(minutes=120)
    first_city = planner.geo["cities"][0]["id"]
    changes = []
    for edge in data.get("transport", []):
        option = chosen_option(edge)
        if (edge.get("scope") == "intercity" and edge.get("origin_id") == first_city
            and edge.get("date") == incoming["arrival_time"][:10]
            and option.get("status") == "estimate" and option.get("mode") == "car"
            and not option.get("planning_departure_time") and not option.get("departure_time")):
            default = datetime.fromisoformat(edge["date"] + "T09:00")
            if default < earliest:
                changes.append({"id": edge["id"], "date": earliest.date().isoformat(), "departure_time": earliest.strftime("%H:%M")})
    if changes:
        apply_adjustments(planner, data, workspace, {"transfer_changes": changes})


def apply_adjustments(planner, data, workspace, args):
    transfers = copy.deepcopy(data.get("transport", []))
    for change in args.get("transfer_changes", []):
        edge = next((t for t in transfers if t.get("id") == change["id"]), None)
        if not edge:
            raise ValueError(f"交通段不存在：{change['id']}")
        option = next((o for o in edge.get("options", []) if o.get("status") == "estimate" and o.get("mode") == "car"), None)
        if not option:
            raise ValueError("只能调整已有道路估算的计划出发时刻，不能改写真实航班/列车/巴士班次；请选择有来源的替代方案")
        duration = option.get("duration_minutes") or option.get("planning_duration_minutes")
        if not duration:
            raise ValueError("缺少实查道路耗时，先查询路线")
        depart = datetime.fromisoformat(change["date"] + "T" + change["departure_time"])
        if not planner.trip.travel_dates.start_date <= depart.date() <= planner.trip.travel_dates.end_date:
            raise ValueError("计划转移日期超出旅行日期")
        arrive = depart + timedelta(minutes=float(duration))
        option.update(planning_departure_date=depart.date().isoformat(), planning_departure_time=depart.strftime("%H:%M"),
                      planning_arrival_date=arrive.date().isoformat(), planning_arrival_time=arrive.strftime("%H:%M"),
                      planning_time_basis="自主安排的道路接送时间，按实查道路耗时推算抵达；车辆和实时拥堵未确认")
        for item in edge["options"]:
            item["baseline"] = item is option
    stays = dict(workspace.get("stay_overrides", {}))
    for change in args.get("stay_changes", []):
        city = canonical_city(planner.geo, change["city"])
        if not city or not 1 <= change["day"] < planner.trip.duration_days:
            raise ValueError("住宿调整须使用现有城市及有效住宿日")
        stays[str(change["day"])] = city["name"]
    data["transport"] = transfers
    workspace["stay_overrides"] = stays


def departure(option, edge):
    return option.get("planning_departure_date") or option.get("departure_date") or edge["date"], (
        option.get("planning_departure_time") or option.get("departure_time") or "09:00"
    )[-5:]


def arrival(option, edge):
    when, clock = departure(option, edge)
    if option.get("planning_arrival_date"):
        return option["planning_arrival_date"], option["planning_arrival_time"]
    if option.get("arrival_time"):
        return option.get("arrival_date") or when, option["arrival_time"][-5:]
    duration = option.get("duration_minutes") or option.get("planning_duration_minutes")
    if duration:
        value = datetime.fromisoformat(when + "T" + clock) + timedelta(minutes=float(duration))
        return value.date().isoformat(), value.strftime("%H:%M")
    return when, ""
