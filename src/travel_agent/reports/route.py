"""Canonical destination identities, stays and stage-local evidence coverage."""

from __future__ import annotations

import copy
import hashlib
from datetime import timedelta

from travel_agent.reports.planning import coord, records
from travel_agent.reports.route_contract import (
    ApprovedRouteConflict,
    city_keys,
    fit_visit_counts,
    key,
    reserve_kinds,
    route_issues,
)


def identity(country, query):
    return country.lower() + ":" + hashlib.sha256(query.casefold().encode()).hexdigest()[:12]


def preserve_route_notes(geo):
    """Allocation-time opinions are history, not claims about a finished itinerary."""
    notes = geo.pop("tradeoffs", [])
    if isinstance(notes, str):
        notes = [notes]
    geo["planning_notes"] = list(dict.fromkeys([*geo.get("planning_notes", []), *notes]))
    return geo


def calendar_cities(geo, contract, total_days):
    cities = copy.deepcopy(geo.get("cities") or [])
    if not cities:
        return cities
    for city in cities:
        city["days"] = int(city.get("days") or 0)
        # Calendar reservations are produced by the runtime, not free-form model output.
        for field in ("reserve_before", "reserve_after"):
            city.pop(field, None)
    reserved = (contract or {}).get("reserved", {})
    fit_visit_counts(contract, cities, total_days - sum(reserved.values()))
    if n := reserved.get("outbound"):
        cities[0]["reserve_before"] = ["outbound"] * n
        cities[0]["days"] += n
    if n := reserved.get("return"):
        cities[-1]["reserve_after"] = ["return"] * n
        cities[-1]["days"] += n
    if n := reserved.get("flex"):
        # The model can specify a route anchor, not turn buffer days into a destination's visits.
        anchor = geo.get("buffer_anchor")
        anchors = cities[:-1] or cities
        city = next((c for c in anchors if isinstance(anchor, str) and key(anchor) in city_keys(c)), anchors[-1])
        city.setdefault("reserve_after", [])[0:0] = ["flex"] * n
        city["days"] += n
    return cities


def resolve_route(planner, previous=None):
    old = previous or {}
    contract = getattr(planner, "route_contract", None)
    value = {
        "query": planner.query,
        "previous_route": {k: v for k, v in old.get("region", {}).items()
                           if k not in {"tradeoffs", "planning_notes"}},
        "available_capabilities": [x["name"] for x in planner.tools.capability_catalog()],
    }
    if contract:
        value["approved_route"] = contract
        value["city_days_target"] = planner.trip.duration_days - sum(contract.get("reserved", {}).values())
        if contract["stops"]:
            # Do not reintroduce the shortened rewrite that dropped the cities.
            # Route selection is based on the approval, not the caller's prose.
            value.pop("query")
    geo = planner.ask(
        "为旅行建立唯一的城市停留计划。根据兴趣选择合理数量的城市，不追求去遍全国家。"
        "尊重明确指定城市及天数；城市按实际通行顺序排列，合计days等于总天数，含抵达和离境日。"
        "name保留用户正确的中文名称，query必须为英文或当地通行名称，aliases含其他中文和当地别名。"
        "修改时保留未要求移除的城市；考虑新增城市时先分析交通代价，在tradeoffs说明建议和取舍。"
        "提供了approved_route时，以该已批准的城市与天数约束为准。query是当前步骤的指令，即使只提到部分城市，"
        "也必须在结果保留approved_route.stops中的全部停留地及其min_days/max_days范围。"
        "如果有city_days_target，cities.days只算当地停留，合计等于city_days_target，不是含独立出发/返程/机动的全程天数。"
        "独立预留日由程序另计；buffer_anchor可给一个已有停留地作为机动时间的落脚区，不增加当地游览天数。"
        "entry_mode/exit_mode从flight/rail/road/none选择，依据距离、国境、用户已有预订与工具覆盖。"
        "constraint_checks只检查城市顺序与停留分配，不能把尚未执行的交通/酒店研究判为违反约束。每城市给两个不超过18字的search_keywords，覆盖用户最感兴趣的体验与路线。constraint_checks只检查城市选择与天数分配，不检查尚未执行的酒店、交通、餐饮检索。不要返回坐标和未经检索的票价。"
        '返回 {"country_code":"ISO两字母","country_name":string,"currency":"ISO币种",'
        '"cities":[{"name":string,"query":string,"country_code":"该城市自己的ISO两字母国家码","aliases":[string],"search_keywords":[string],"days":number}],'
        '"entry_mode":string,"exit_mode":string,"buffer_anchor":string或null,"tradeoffs":[string],'
        '"constraint_checks":[{"scope":"route","requirement":string,"satisfied":boolean,"reason":string}]}。',
        value,
        "region",
    )
    code = str(geo.get("country_code", "")).upper()
    if len(code) != 2 or not geo.get("cities"):
        raise RuntimeError("未识别可核验的旅行城市，需补充目的地。")
    from pycountry import countries

    country = countries.get(alpha_2=code)
    if country:
        geo["country_name"] = country.name
    cities = calendar_cities(geo, contract, planner.trip.duration_days)
    counts = [int(c.get("days") or 0) for c in cities]
    violations = route_issues(contract, cities)
    if violations or any(n <= 0 for n in counts) or sum(counts) != planner.trip.duration_days:
        repaired = planner.ask(
            "修复系统生成的城市停留分配。保留用户明确的城市与约束，days正整数合计必须等于total_days。"
            "approved_route是已批准的约束，优先于query里的临时建议，逐项修复route_violations。"
            "给出了city_days_target时，合计的是当地停留，必须等于city_days_target，独立预留日由程序另计。"
            "不是让用户修复系统算错的天数。返回与原route相同结构，包含全部城市。",
            {**({"query": planner.query} if "query" in value else {}),
             "route": geo, "total_days": planner.trip.duration_days,
             "approved_route": contract, "route_violations": violations,
             "city_days_target": value.get("city_days_target"),
             "validation": {"counts": counts, "sum": sum(counts)}},
            "route_allocation_repair",
        )
        cities = calendar_cities(repaired, contract, planner.trip.duration_days)
        counts = [int(c.get("days") or 0) for c in cities]
        if violations := route_issues(contract, cities):
            raise ApprovedRouteConflict(violations)
        if not cities or any(n <= 0 for n in counts) or sum(counts) != planner.trip.duration_days:
            raise ValueError(f"系统路线分配尚未修复：生成{counts}，要求合计{planner.trip.duration_days}天；需重新分配，不能归因于用户输入。")
        geo.update(repaired)
    geo["cities"] = cities
    start_day = 1
    for city in cities:
        city_code = str(city.get("country_code") or code).upper()
        if not countries.get(alpha_2=city_code):
            raise ValueError(f"城市{city.get('name')}的国家身份尚未核验")
        city.update(
            id=identity(city_code, city["query"]),
            country_code=city_code,
            city=city["name"],
            start_day=start_day,
            end_day=start_day + int(city["days"]) - 1,
        )
        city["city_id"] = city["id"]
        city["aliases"] = list(
            dict.fromkeys([city["name"], city["query"], *city.get("aliases", [])])
        )
        old_city = next(
            (
                c
                for c in old.get("region", {}).get("cities", [])
                if c.get("id") == city["id"] and c.get("center")
            ),
            None,
        )
        if old_city:
            city["center"] = old_city["center"]
        else:
            data = (
                planner.call("amap_maps__maps_geo", {"address": city["name"], "city": city["name"]})
                if city_code == "CN"
                else planner.call(
                    "open_maps__search_places", {"query": city["query"], "country_code": city_code}
                )
            )
            choices = records(data, "geocodes") if city_code == "CN" else data.get("places", [])
            choices = [c for c in choices if coord(c.get("location"))]
            choices.sort(
                key=lambda c: (
                    c.get("osm_value") not in {"city", "town", "administrative", "village"}
                )
            )
            city["center"] = choices[0]["location"] if choices else None
            if not choices:
                planner.issues.append(f"{city['name']}城市中心未核验，将用国家和独立地点身份检索。")
        city["checkin"] = (
            planner.trip.travel_dates.start_date + timedelta(days=start_day - 1)
        ).isoformat()
        city["checkout"] = (
            planner.trip.travel_dates.start_date
            + timedelta(days=min(city["end_day"], planner.trip.duration_days - 1))
        ).isoformat()
        start_day += int(city["days"])
    geo.update(country_code=code, center=cities[0].get("center"), version=3)
    return preserve_route_notes(geo)


def canonical_city(geo, name):
    name = str(name or "").casefold().removesuffix("市")
    return next(
        (
            c
            for c in geo["cities"]
            if name
            in [
                str(a).casefold().removesuffix("市")
                for a in [c["name"], c["query"], *c.get("aliases", [])]
            ]
        ),
        None,
    )


def city_notes(notes, city):
    aliases = [
        str(x).casefold()
        for x in city.get("aliases", [city["name"], city["query"]])
        if len(str(x)) > 1
    ]
    return [
        n
        for n in notes
        if any(a in (n.get("title", "") + " " + n.get("content", "")).casefold() for a in aliases)
    ]


def route_days(planner):
    return [
        {
            "day": i,
            "date": (planner.trip.travel_dates.start_date + timedelta(days=i - 1)).isoformat(),
            "city": city["name"],
            "stay_city": city["name"],
            "city_id": city["id"],
            "area": city["name"],
            **({"approved_reserve_kind": kind} if (kind := reserved_day(city, i)) else {}),
        }
        for city in planner.geo["cities"]
        for i in range(city["start_day"], city["end_day"] + 1)
    ]


def reserved_day(city, day):
    before, after = city.get("reserve_before", []), city.get("reserve_after", [])
    offset = day - city["start_day"]
    if 0 <= offset < len(before):
        return before[offset]
    offset = day - city["end_day"] + len(after) - 1
    return after[offset] if 0 <= offset < len(after) else None


def chosen_option(transport):
    options = transport.get("options", [])
    return next((o for o in options if o.get("baseline")), next(iter(options), {}))


def external_windows(transports):
    directions = [d for t in transports for d in chosen_option(t).get("directions", [])]
    for transport in transports:
        if transport.get("scope") == "external" and transport.get("kind") in {"entry", "exit"}:
            option = chosen_option(transport)
            direction = {
                **option, "direction": "outbound" if transport["kind"] == "entry" else "return",
                "origin": transport["origin"], "destination": transport["destination"],
            }
            for label in ("departure", "arrival"):
                clock = option.get(f"planning_{label}_time") or option.get(f"{label}_time")
                when = option.get(f"planning_{label}_date") or option.get(f"{label}_date")
                direction[f"{label}_time"] = (
                    f"{when}T{clock}" if when and clock and len(clock) <= 8
                    else clock if clock and "T" in clock else ""
                )
            directions.append(direction)
    return next((d for d in directions if d.get("direction") == "outbound"), {}), next(
        (d for d in directions if d.get("direction") == "return"), {}
    )


def stays_for_days(planner, days, transports):
    """Allocate nights from actual local arrival; a departure-only day has no hotel."""
    arrival, departure = external_windows(transports)
    stays = []
    for day in days[:-1]:
        if day.get("no_hotel"):
            continue
        if arrival.get("arrival_time", "")[:10] and day["date"] < arrival["arrival_time"][:10]:
            continue
        if (
            departure.get("departure_time", "")[:10]
            and day["date"] >= departure["departure_time"][:10]
        ):
            continue
        city = canonical_city(planner.geo, day.get("stay_city") or day["city"])
        if not city:
            continue
        if not stays or stays[-1]["city_id"] != city["id"] or stays[-1]["end_day"] + 1 != day["day"]:
            stays.append(
                {
                    **copy.deepcopy(city),
                    "start_day": day["day"],
                    "end_day": day["day"],
                    "checkin": day["date"],
                }
            )
        stays[-1]["end_day"] = day["day"]
        stays[-1]["checkout"] = (
            planner.trip.travel_dates.start_date + timedelta(days=day["day"])
        ).isoformat()
    return stays


def upgrade_snapshot(previous):
    """Add identities locally; a targeted edit does not force a new full research run."""
    old = copy.deepcopy(previous)
    if not old or old.get("research_version") == 3 or not old.get("region"):
        return old
    geo = old["region"]
    original = geo.get("cities", [])
    groups = []
    for day in old.get("days", []):
        if not groups or groups[-1]["name"] != day["city"]:
            found = next((c for c in original if c["name"] == day["city"]), {})
            query = found.get("query") or day["city"]
            groups.append(
                {
                    **found,
                    "name": day["city"],
                    "query": query,
                    "city": day["city"],
                    "id": identity(geo["country_code"], query),
                    "city_id": identity(geo["country_code"], query),
                    "country_code": geo["country_code"],
                    "aliases": [day["city"], query],
                    "start_day": day["day"],
                    "end_day": day["day"],
                    "days": 0,
                }
            )
        groups[-1]["days"] += 1
        groups[-1]["end_day"] = day["day"]
        day["city_id"] = groups[-1]["id"]
        if not day.get("stops"):
            day["research_gap"] = True
    geo["cities"] = groups or original
    for p in old.get("places", []):
        city = canonical_city(geo, p.get("city"))
        if city:
            p["city_id"] = city["id"]
        p.setdefault("kind", "sightseeing")
    for t in old.get("transport", []):
        t.setdefault("scope", "external" if t.get("type") == "往返航班" else "intercity")
        t.setdefault("id", f"legacy:{t['origin']}:{t['destination']}:{t.get('date')}")
    geo.setdefault(
        "entry_mode",
        "flight" if any(t.get("type") == "往返航班" for t in old.get("transport", [])) else "rail",
    )
    return old


def fit_route_to_flights(planner, external):
    """Reconcile proposed city slots with actual flight dates before querying local legs."""
    arrival, departure = external_windows(external)
    if not arrival or not departure or len(planner.geo["cities"]) < 2:
        return False
    old = copy.deepcopy(planner.geo["cities"])
    contract = getattr(planner, "route_contract", None)
    decision = planner.ask(
        "用已查询的真实航班时刻复核多城市路线，可调整各城市日历天数或建议不加入可选城市。"
        "days表示旅行日历槽，不等于当地完整游览天数；抵达前不能游览，凌晨离境需前一天回到出境城市。"
        "用户指定某城市玩两天时，需要真正留出当地游览窗口，不能把尚未抵达的飞行日算进去。"
        "根据地点间坐标、具体体验建议时长、偏好与总天数判断是否过于奔波。不能编造交通时长或票价。"
        "保留用户明确必去的城市；对询问是否值得加入的城市，可以建议不加入并解释，不能无解释删除其他城市。"
        "approved_route中已批准的城市与天数范围不能改。遇到真实航班时间冲突而没有符合约束的排法，保留原路线并在issues给出依据。"
        "以下城市days含独立预留日；停留约束只计算days减reserved_days，任何预留日均不可以被改成游览或去掉。"
        "不得新增未给出的城市，首个入境城市必须保留。返回的城市days正整数合计等于总天数。"
        "若改动不能解决冲突则保持路线并在issues说明，不伪称已通过。"
        '返回 {"cities":[{"id":已有id,"days":number}],"tradeoffs":[string],"issues":[string]}。',
        {
            "query": planner.query,
            "prior_request": planner.previous.get("revision", {}).get("query"),
            "approved_route": contract,
            "constraints": planner.trip.hard_constraints,
            "total_days": planner.trip.duration_days,
            "arrival": {
                k: arrival.get(k)
                for k in ("origin", "destination", "departure_time", "arrival_time")
            },
            "departure": {
                k: departure.get(k)
                for k in ("origin", "destination", "departure_time", "arrival_time")
            },
            "cities": [{**{k: c.get(k) for k in ("id", "name", "days", "center")},
                        "reserved_days": len(reserve_kinds(c))} for c in old],
        },
        "route_feasibility",
    )
    lookup = {c["id"]: c for c in old}
    values = decision.get("cities", [])
    if (
        not values
        or values[0].get("id") != old[0]["id"]
        or any(v.get("id") not in lookup or int(v.get("days") or 0) <= 0 for v in values)
    ):
        return False
    if sum(int(v["days"]) for v in values) != planner.trip.duration_days:
        return False
    cities = []
    first = 1
    for value in values:
        city = copy.deepcopy(lookup[value["id"]])
        n = int(value["days"])
        city.update(
            days=n,
            start_day=first,
            end_day=first + n - 1,
            checkin=(planner.trip.travel_dates.start_date + timedelta(days=first - 1)).isoformat(),
            checkout=(
                planner.trip.travel_dates.start_date
                + timedelta(days=min(first + n - 1, planner.trip.duration_days - 1))
            ).isoformat(),
        )
        cities.append(city)
        first += n
    if violations := route_issues(contract, cities):
        raise ApprovedRouteConflict(violations)
    planner.geo["cities"] = cities
    preserve_route_notes(planner.geo)
    planner.geo["planning_notes"] = list(dict.fromkeys([
        *planner.geo["planning_notes"], *decision.get("tradeoffs", []),
    ]))
    planner.issues.extend(decision.get("issues", []))
    return [(c["id"], c["days"]) for c in cities] != [(c["id"], c["days"]) for c in old]
