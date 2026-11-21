"""Schedule sourced experiences into local-day windows, meals and transfer blocks."""

from __future__ import annotations

import copy

from travel_agent.reports.adjustments import arrival as transfer_arrival
from travel_agent.reports.adjustments import departure as transfer_departure
from travel_agent.reports.planning import km
from travel_agent.reports.route import chosen_option, external_windows, route_days


def hhmm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def minute(iso, default=0):
    try:
        return int(iso[11:13]) * 60 + int(iso[14:16])
    except (ValueError, TypeError):
        return default


def before_transfers(planner, day, points, transport, used, start):
    """Use sourced experiences in genuine windows between same-day city transfers."""
    edges = []
    for edge in transport:
        option = chosen_option(edge)
        when, clock = transfer_departure(option, edge)
        if edge.get("scope") == "intercity" and when == day["date"]:
            edges.append((clock, edge, option))
    edges.sort(key=lambda row: row[0])
    selected = []
    cursor, city_id = start, day["city_id"]
    for clock, edge, option in edges:
        previous_count = len(selected)
        until = minute("2000-01-01T" + clock) - 30
        candidates = [p for p in points if p.get("city_id") == edge.get("origin_id") == city_id
                      and p["id"] not in used and p.get("kind") in {"sightseeing", "activity", "shop"}]
        candidates.sort(key=lambda p: p.get("kind") != "activity")
        for point in candidates:
            duration = int(point.get("duration_minutes") or 120)
            if cursor + duration > until or len(selected) >= 2:
                continue
            leg = planner.leg(selected[-1], point) if selected and selected[-1].get("city_id") == city_id else None
            if leg and not leg.get("verified"):
                continue
            travel = int(leg.get("minutes") or 0) + 15 if leg else 0
            if cursor + travel + duration > until:
                continue
            cursor += travel
            block = {**copy.deepcopy(point), "type": point.get("kind", "sightseeing"),
                     "start_time": hhmm(cursor), "end_time": hhmm(cursor + duration),
                     "duration_minutes": duration, "status": "sourced"}
            selected.append(block)
            day["timeline"].append(block)
            used.add(point["id"])
            cursor += duration
        if len(selected) > previous_count and 11 * 60 <= cursor <= 14 * 60 and max(cursor, 12 * 60) + 45 <= until:
            meals = [p for p in points if p.get("city_id") == city_id and p.get("kind") in {"meal", "cafe"}][:2]
            day["timeline"].append({"type": "meal", "name": "转移前午餐与休息", "start_time": hhmm(max(cursor, 12 * 60)),
                                    "duration_minutes": 45, "alternatives": copy.deepcopy(meals),
                                    "reason": "在当前片区用餐后再跨城，店铺为来源候选。" if meals else "跨城前预留简餐时间，具体店铺现场选择，费用纳入餐饮估算。",
                                    "status": "sourced" if meals else "unavailable"})
        arriving_date, arriving_clock = transfer_arrival(option, edge)
        if arriving_date == day["date"] and arriving_clock:
            cursor = max(start, minute("2000-01-01T" + arriving_clock) + 90)
            city_id = edge.get("destination_id")
    return selected


def build_days(planner, points, transport):
    days = route_days(planner)
    arrival, departure = external_windows(transport)
    used = set()
    previous = {d["day"]: d for d in getattr(planner, "previous_scheduled_days", [])}
    for day_index, day in enumerate(days):
        if day["day"] in previous:
            day.update(copy.deepcopy(previous[day["day"]]))
            used.update(b.get("id") for b in day.get("timeline", []) if b.get("id"))
            continue
        day.update(
            stops=[],
            legs=[],
            timeline=[],
            food=[],
            title=f"{day['city']} · 当地探索",
            intro="按片区安排，并预留用餐和接驳时间。",
        )
        start, end = 9 * 60, 20 * 60
        airport_deadline, latest_transfer_arrival = None, None
        if arrival.get("arrival_time"):
            when = arrival["arrival_time"][:10]
            if day["date"] < when:
                day.update(
                    type="transfer",
                    title="出发与途中转机",
                    intro="尚未抵达目的地，今天安排长途交通。",
                    stay_city=None,
                    no_hotel=True,
                )
                day["timeline"] = [
                    {
                        "type": "transfer",
                        "name": f"前往{arrival.get('destination', day['city'])}",
                        "start_time": arrival.get("departure_time", "")[11:16],
                        "reason": f"当地抵达时间 {arrival['arrival_time']}，抵达前不安排游览。",
                        "status": "quote",
                    }
                ]
                day["exception_reason"] = "国际航班及转机日"
                if getattr(planner, "on_days", None):
                    planner.on_days(days[:day_index + 1])
                continue
            if day["date"] == when:
                day["transfer_day"] = True
                start = max(start, minute(arrival["arrival_time"]) + 120)
        if departure.get("departure_time") and day["date"] == departure["departure_time"][:10]:
            day["transfer_day"] = True
            prior_stay = days[day_index - 1].get("stay_city") if day_index else None
            if prior_stay and not any(e.get("scope") == "intercity" and transfer_departure(chosen_option(e), e)[0] == day["date"] for e in transport):
                from travel_agent.reports.route import canonical_city
                location = canonical_city(planner.geo, prior_stay)
                if location:
                    day.update(city=location["name"], city_id=location["id"], area=location["name"], stay_city=None, no_hotel=True)
            departure_minute = minute(departure["departure_time"])
            end = max(0, min(end, departure_minute - 180))
            airport_deadline = end
            if departure_minute < 6 * 60:
                entry_city = planner.geo["cities"][0]
                day.update(
                    city=entry_city["name"],
                    city_id=entry_city["id"],
                    area=entry_city["name"],
                    stay_city=None,
                )
                day.update(
                    type="transfer",
                    title="夜间离境与返程",
                    intro="前一日已返回出境城市，夜间办理手续并返程；今天不安排游览。",
                    no_hotel=True,
                )
                day["exception_reason"] = "凌晨返程航班"
                day["timeline"] = [
                    {
                        "type": "transfer",
                        "name": "返程航班",
                        "start_time": departure["departure_time"][11:16],
                        "reason": f"{departure.get('origin', '')} → {departure.get('destination', '')}；抵达 {departure.get('arrival_time', '待核实')}",
                    }
                ]
                if getattr(planner, "on_days", None):
                    planner.on_days(days[:day_index + 1])
                continue
        for edge in transport:
            if (
                edge.get("scope") == "external"
                and edge.get("type") == "高铁"
                and edge.get("date") == day["date"]
            ):
                option = chosen_option(edge)
                day["transfer_day"] = True
                if edge.get("destination") == day["city"] and option.get("arrive_time"):
                    start = max(start, minute("2000-01-01T" + option["arrive_time"]) + 60)
                elif edge.get("origin") == day["city"] and option.get("start_time"):
                    end = min(end, minute("2000-01-01T" + option["start_time"]) - 60)
        reserved = day.get("approved_reserve_kind")
        preselected = [] if reserved else before_transfers(planner, day, points, transport, used, start)
        in_transit = False
        for edge in transport:
            if edge.get("scope") != "intercity":
                continue
            option = chosen_option(edge)
            departure_date, departure_clock = transfer_departure(option, edge)
            arrival_date, arrival_clock = transfer_arrival(option, edge)
            if not departure_date <= day["date"] <= arrival_date:
                continue
            duration = option.get("duration_minutes") or option.get("planning_duration_minutes")
            if (arrival.get("arrival_time") and day["date"] == arrival["arrival_time"][:10]
                and edge.get("origin_id") == planner.geo["cities"][0]["id"]
                and minute("2000-01-01T" + departure_clock) < minute(arrival["arrival_time"]) + 120):
                day["timing_warning"] = f"入境航班{arrival['arrival_time'][-5:]}抵达，需留2小时办理手续及接驳；当前{departure_clock}的城际出发无法衔接，请调整接送时刻或班次。"
            block = {
                "type": "transfer",
                "name": f"{edge['origin']} → {edge['destination']}",
                "start_time": departure_clock if day["date"] == departure_date else "00:00",
                "reason": (option.get("route") or edge.get("raw_summary", "城际交通需核实"))
                + (f"；抵达 {arrival_date} {arrival_clock}" if arrival_clock else "")
                + (
                    "；" + option["planning_duration_basis"]
                    if option.get("planning_duration_basis")
                    else ""
                ),
                "source_url": option.get("url") or option.get("source_url") or edge.get("url", ""),
                "duration_minutes": duration if arrival_date == departure_date else None,
                "status": edge.get("status", "reference"),
            }
            day["timeline"].append(block)
            day["transfer_day"] = True
            if day["date"] < arrival_date:
                in_transit = True
                day["no_hotel"] = True
                continue
            # Activity windows use the actual local arrival, never origin clock + flight duration.
            if arrival_clock:
                latest_transfer_arrival = max(latest_transfer_arrival or 0, minute("2000-01-01T" + arrival_clock))
                start = max(start, minute("2000-01-01T" + arrival_clock) + 90)
            elif duration:
                start = max(
                    start, minute("2000-01-01T" + departure_clock, 9 * 60) + int(duration) + 90
                )
            else:
                day["transfer_unresolved"] = True
                start = max(start, 14 * 60)
            if edge.get("destination_id") and edge["destination_id"] != day["city_id"]:
                day["closing_transfer"] = True
                day.update(
                    city=edge["destination"],
                    city_id=edge["destination_id"],
                    stay_city=edge["destination"],
                    area=edge["destination"],
                )
        if in_transit:
            day.update(
                type="transfer",
                title="城际转移与途中休息",
                intro="该段跨越日期，抵达前不安排目的地游览；途中住宿不计入酒店夜晚。",
            )
            day["exception_reason"] = "跨日城际交通"
            day["stops"] = preselected
            day["timeline"].sort(key=lambda b: b.get("start_time", "00:00"))
            if getattr(planner, "on_days", None):
                planner.on_days(days[:day_index + 1])
            continue
        day["available_from"], day["available_until"] = hhmm(start), hhmm(end)
        if airport_deadline is not None and latest_transfer_arrival is not None and latest_transfer_arrival > airport_deadline:
            day["timing_warning"] = (
                f"当前接送预计{hhmm(latest_transfer_arrival)}抵达，但航班{departure.get('departure_time', '')[-5:]}需至少提前3小时、"
                f"即{hhmm(airport_deadline)}前到达机场；请将道路接送至少提前{latest_transfer_arrival - airport_deadline}分钟，不能只改说明文字。"
            )
        if reserved:
            title = {"outbound": "出发与抵达", "return": "返程", "flex": "机动与休整"}[reserved]
            day.update(type="transfer" if reserved != "flex" or day["timeline"] else "rest",
                       title=title, intro="这是已批准的预留日，不计入城市游览天数。", exception_reason=title)
            if getattr(planner, "on_days", None):
                planner.on_days(days[:day_index + 1])
            continue
        if end - start < 90:
            day.update(
                type="transfer",
                title=f"{day['city']} · 抵离与接驳",
                intro="主要时间用于航班、转机和接驳，保留休息缓冲。",
            )
            day["exception_reason"] = "抵离日可用游览时间不足"
            day["stops"] = preselected
            if preselected:
                day["title"] = " · ".join(p["name"] for p in preselected) + "与跨城转移"
            if not day["timeline"]:
                day["timeline"].append(
                    {
                        "type": "transfer",
                        "name": "机场接驳及办理手续",
                        "start_time": hhmm(max(0, end)),
                        "reason": "按已检索航班时刻预留接驳，不强行加入景点。",
                    }
                )
            day["timeline"].sort(key=lambda b: b.get("start_time", "00:00"))
            if getattr(planner, "on_days", None):
                planner.on_days(days[:day_index + 1])
            continue
        city_points = [p for p in points if p.get("city_id") == day["city_id"]]
        overnight_ids = getattr(planner, "overnight_place_ids", set())
        overnight = next((
            p for p in city_points if p["id"] in overnight_ids and p["id"] not in used
            and isinstance(p.get("duration_minutes"), int | float) and p["duration_minutes"] > 0
        ), None)
        night_start = None
        if (overnight and not day.get("no_hotel") and not day.get("closing_transfer")
            and str(day["day"]) not in getattr(planner, "stay_overrides", {})
            and day_index + 1 < len(days) and days[day_index + 1]["city_id"] == day["city_id"]):
            proposed = min(17 * 60, 33 * 60 - int(overnight["duration_minutes"]))
            if max(start + 60, 15 * 60) <= proposed <= end:
                night_start = proposed
                end = proposed - 120
        primary = [
            p
            for p in city_points
            if p.get("kind", "sightseeing") in {"sightseeing", "activity", "shop"}
            and p["id"] not in used and p["id"] not in overnight_ids
        ]
        primary.sort(
            key=lambda p: (
                p["id"] not in getattr(planner, "mandatory_place_ids", set()),
                p.get("kind") == "shop",
                p.get("kind") == "activity",
            )
        )
        remaining_days = sum(
            1 for d in days if d["city_id"] == day["city_id"] and d["day"] >= day["day"]
        )
        daily_target = min(3, max(1, (len(primary) + remaining_days - 1) // remaining_days))
        selected = []
        cursor = start
        lunch_reserved = start >= 13 * 60 or end < 12 * 60
        while primary and len(selected) < daily_target:
            point = primary[0] if not selected else min(primary, key=lambda p: km(p, selected[-1]))
            if selected and km(point, selected[-1]) > 80:
                break
            duration = max(30, min(480, int(point.get("duration_minutes") or 120)))
            leg = planner.leg(selected[-1], point) if selected else None
            if leg and not leg.get("verified"):
                # Unreachable combinations are regrouped; no invented 45-minute road leg.
                primary.remove(point)
                continue
            travel = (leg.get("minutes") or 45) + 15 if leg else 0
            # Fit meals explicitly. Consecutive activity times cannot overlap them.
            if cursor < 13 * 60 and cursor + travel + duration > 12 * 60:
                duration_before_lunch = 12 * 60 - cursor - travel
                if duration_before_lunch >= 60 and duration <= 180:
                    duration = min(duration, duration_before_lunch)
                elif cursor >= 11 * 60:
                    cursor = max(cursor, 13 * 60)
                    lunch_reserved = True
            if cursor + travel + duration > end:
                primary.remove(point)
                continue
            cursor += travel
            block = {
                **copy.deepcopy(point),
                "type": point.get("kind", "sightseeing"),
                "start_time": hhmm(cursor),
                "duration_minutes": duration,
                "end_time": hhmm(cursor + duration),
                "status": "sourced",
            }
            selected.append(block)
            day["timeline"].append(block)
            used.add(point["id"])
            primary.remove(point)
            if leg:
                day["legs"].append(leg)
                selected[-2]["route_leg"] = leg
            cursor += duration
            if not lunch_reserved and cursor >= 12 * 60:
                cursor = max(13 * 60, cursor + 60)
                lunch_reserved = True
            if len(selected) >= 2 and cursor >= 17 * 60:
                break
        if night_start is not None:
            prior = (selected or preselected)[-1] if selected or preselected else None
            leg = planner.leg(prior, overnight) if prior else None
            transit = (leg.get("minutes") or 0) + 15 if leg else 0
            if (not leg or leg.get("verified")) and cursor + transit <= night_start:
                total = int(overnight["duration_minutes"])
                arrival_next_day = night_start + total - 24 * 60
                if arrival_next_day >= 0:
                    block = {**copy.deepcopy(overnight), "type": "activity", "start_time": hhmm(night_start),
                             "duration_minutes": total, "end_time": hhmm(arrival_next_day), "status": "sourced",
                             "end_date": days[day_index + 1]["date"],
                             "reason": (overnight.get("reason") or "") + " 结束时间为次日建议时刻。"
                             "预订前核实运营方、当日条件和接送；此晚按体验过夜计，酒店预算不重复计算。"}
                    selected.append(block)
                    day["timeline"].append(block)
                    day["no_hotel"] = True
                    used.add(overnight["id"])
                    if leg:
                        day["legs"].append(leg)
                        prior["route_leg"] = leg
        day["stops"] = [s for s in [*preselected, *selected] if s["type"] in {"sightseeing", "activity", "shop"}]
        if day.get("closing_transfer") and departure.get("departure_time"):
            dep_date = departure["departure_time"][:10]
            if day["date"] < dep_date and minute(departure["departure_time"]) < 6 * 60:
                day["no_hotel"] = True
                day["intro"] += " 晚间接驳机场并候机，不计当晚酒店。"
        if not selected and day.get("closing_transfer"):
            day.update(
                type="transfer",
                title=f"返回{day['city']}，接驳与候机",
                intro="返回出境城市，保留接驳、休息和机场办理手续时间。",
            )
            day["exception_reason"] = "离境前的城际接驳日"
        elif not selected:
            day["research_gap"] = True
            day["intro"] = (
                "当前城市尚缺可定位、符合偏好的具体体验，需继续检索；不把泛泛描述视为完成。"
            )
        else:
            day["title"] = " · ".join(s["name"] for s in day["stops"][:2])
        add_meals(planner, day, city_points, start, end, used)
        if day["stops"] and day["city_id"] in getattr(planner, "friend_time_cities", set()):
            for t in range(max(start, 13 * 60), end - 119, 30):
                if all(t + 120 <= (b_start := minute("2000-01-01T" + b.get("start_time", "00:00")))
                       or t >= b_start + int(b.get("duration_minutes") or 0) for b in day["timeline"]):
                    day["timeline"].append({"type": "friend_time", "name": "预留会友时间",
                                            "start_time": hhmm(t), "end_time": hhmm(t + 120), "duration_minutes": 120,
                                            "reason": "按你的访友安排留出时间，见面的地点和时段由你们另行商定。",
                                            "status": "planned"})
                    break
        day["timeline"].sort(key=lambda b: b.get("start_time") or "99:99")
        if getattr(planner, "on_days", None):
            planner.on_days(days[:day_index + 1])
    return days


def add_meals(planner, day, points, start, end, used):
    for time, label in [(12 * 60, "午餐"), (18 * 60, "晚餐")]:
        if not start <= time or time + 45 > end:
            continue
        # Long visits keep their full duration; meal slot follows the actual visit.
        actual = time
        for block in sorted(day["timeline"], key=lambda b: b.get("start_time", "00:00")):
            a = minute("2000-01-01T" + block.get("start_time", "00:00"))
            b = a + int(block.get("duration_minutes") or 0)
            if actual < b and actual + 45 > a:
                actual = b
        if actual + 45 > (end if end < 20 * 60 else 22 * 60):
            continue
        cafes = [p for p in points if p.get("kind") in {"meal", "cafe"}]
        cafes.sort(key=lambda p: p["id"] in used)
        anchors = day["stops"]
        if anchors:
            cafes.sort(key=lambda p: (p["id"] in used, min(km(p, a) for a in anchors)))
        cafes = [p for p in cafes if not anchors or min(km(p, a) for a in anchors) <= 12][:2]
        if cafes:
            # Alternatives form a choice, not sequential double meals or duplicate costs.
            used.add(cafes[0]["id"])
            day["timeline"].append(
                {
                    "type": "meal",
                    "name": label,
                    "start_time": hhmm(actual),
                    "duration_minutes": 45,
                    "reason": "按当天片区选择一家；以下为候选。",
                    "alternatives": copy.deepcopy(cafes),
                    "status": "sourced",
                }
            )
            day["food"].extend(p["name"] for p in cafes)
        else:
            day["timeline"].append(
                {
                    "type": "meal",
                    "name": label + "与休息",
                    "start_time": hhmm(actual),
                    "duration_minutes": 45,
                    "reason": "具体店铺尚缺可核验推荐，预留就餐时间，费用计入餐饮估算。",
                    "status": "unavailable",
                }
            )
    cafes = [p for p in points if p.get("kind") == "cafe" and p["id"] not in used]
    if cafes and day["stops"]:
        p = min(cafes, key=lambda p: min(km(p, s) for s in day["stops"]))
        blocks = day["timeline"]
        for t in range(max(start, 14 * 60), min(end - 30, 18 * 60), 30):
            if all(
                t + 30 <= minute("2000-01-01T" + b.get("start_time", "00:00"))
                or t
                >= minute("2000-01-01T" + b.get("start_time", "00:00"))
                + int(b.get("duration_minutes") or 0)
                for b in blocks
            ):
                if min(km(p, s) for s in day["stops"]) <= 5:
                    day["timeline"].append(
                        {
                            **copy.deepcopy(p),
                            "type": "cafe",
                            "start_time": hhmm(t),
                            "duration_minutes": 30,
                            "reason": p["reason"],
                            "status": "sourced",
                        }
                    )
                    used.add(p["id"])
                break


def sparse_tour_days(days, transport=()):
    """A normal full day needs a substantive sourced experience, not one short shop stop."""
    boundary_dates = set()
    for edge in transport:
        if edge.get("scope") == "external":
            boundary_dates.add(edge.get("date"))
            for direction in chosen_option(edge).get("directions", []):
                boundary_dates.update(
                    str(direction.get(k, "") or "")[:10] for k in ("departure_time", "arrival_time")
                )
    gaps = []
    for day in days:
        if (
            day.get("type") in {"transfer", "rest"}
            or day.get("transfer_day")
            or day["date"] in boundary_dates
        ):
            continue
        start = minute("2000-01-01T" + str(day.get("available_from") or "09:00"))
        end = minute("2000-01-01T" + str(day.get("available_until") or "20:00"))
        if end - start < 360:
            continue
        blocks = day.get("timeline") or day.get("stops", [])
        if any(b.get("type") == "friend_time" and (b.get("duration_minutes") or 0) >= 120 for b in blocks):
            continue
        meaningful = sum(
            int(b.get("duration_minutes") or 0)
            for b in blocks
            if (b.get("type") or b.get("kind") or "sightseeing")
            in {"sightseeing", "activity", "shop", "cafe"}
            and b.get("source_url")
        )
        if meaningful < 120:
            gaps.append(day)
    return gaps
