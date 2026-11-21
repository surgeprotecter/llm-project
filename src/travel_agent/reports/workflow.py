"""Live full-trip research and dependency-aware revisions of saved reports."""

from __future__ import annotations

import copy
import itertools
import json
import math
import re
import unicodedata
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode, urlsplit

from travel_agent.reports.planning import ResearchBase, coord, km, main_pois, number, records
from travel_agent.reports.quotes import compact, visible_hotel_quote


def plain(value):
    return re.sub(
        r"[^a-z0-9\u4e00-\u9fff]",
        "",
        "".join(
            c
            for c in unicodedata.normalize("NFKD", str(value).lower())
            if not unicodedata.combining(c)
        ),
    )


def matches_base_area(base, place):
    """Reject a distant map match unless the source names that outlying locality."""
    center = (base or {}).get("center")
    if not coord(center) or not coord(place.get("location")):
        return True
    if km({"location": center}, place) <= 150:
        return True
    locality = plain(re.sub(r"\s+city$", "", str(place.get("actual_city") or place.get("city") or ""), flags=re.I))
    evidence_names = [place.get(key) for key in ("name", "mention", "query_name")]
    evidence_names.extend(place.get("aliases") or [])
    return len(locality) >= 4 and any(locality in plain(item) for item in evidence_names if item)


def quote_age(value):
    if not value:
        return "历史检索时间未记录"
    when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (
        "近期检索候选"
        if datetime.now(UTC) - when < timedelta(minutes=30)
        else "历史报价，使用前需刷新"
    )


def dedupe_food(entries):
    grouped = {}
    for item in entries:
        key = item.get("id") or plain(item["name"])
        if key not in grouped:
            grouped[key] = copy.deepcopy(item)
            continue
        first = grouped[key]
        first["note_ids"] = list(
            dict.fromkeys([*first.get("note_ids", []), *item.get("note_ids", [])])
        )
        first["aliases"] = list(
            dict.fromkeys([*first.get("aliases", []), item["name"], *item.get("aliases", [])])
        )
        first["recommended_items"] = list(
            dict.fromkeys([*first.get("recommended_items", []), *item.get("recommended_items", [])])
        )
    return list(grouped.values())


class StandardTripPlanner(ResearchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        saved = self.db.latest_report(self.trip.trip_id, structured_only=True)
        self.previous = json.loads(saved["payload_json"]).get("report", {}) if saved else {}
        self.base_version = saved["version"] if saved else 0
        self.geo = {}
        self.transport_options = []
        self.hotel_targets = []

    def region(self):
        if not self.geo:
            from travel_agent.reports.route import resolve_route

            self.geo = resolve_route(self, self.previous)
        return self.geo

    def extra_notes(self, keywords, existing, limit=4):
        notes = copy.deepcopy(existing)
        seen = {n["id"] for n in notes}
        feeds = {}
        for word in keywords[:3]:
            data = self.call("xiaohongshu__search_feeds", {"keyword": word[:30]})
            for feed in records(data, "feeds")[:6]:
                key = str(feed.get("id") or "")
                if key and key not in seen and (feed.get("xsecToken") or feed.get("xsec_token")):
                    feeds[key] = feed
        shortlist = [
            {"id": k, "title": (f.get("noteCard") or {}).get("displayTitle", "")}
            for k, f in feeds.items()
        ]
        selected = (
            self.ask(
                '选择与本轮补查目标最相关且正文丰富的笔记，优先多地点路线，不能全选单店照片或纯图片无正文的帖子，返回 {"ids":[string]}。',
                {"keywords": keywords, "candidates": shortlist},
                "supplement_notes",
            )
            if shortlist
            else {}
        )
        ids = list(dict.fromkeys([*selected.get("ids", []), *feeds]))
        for key in [i for i in ids if i in feeds][:limit]:
            feed = feeds[key]
            data = self.call(
                "xiaohongshu__get_feed_detail",
                {
                    "feed_id": key,
                    "xsec_token": feed.get("xsecToken") or feed.get("xsec_token"),
                    "load_all_comments": False,
                },
            )
            note = data.get("data", data).get("note", {})
            if note.get("desc"):
                notes.append(
                    {
                        "id": key,
                        "title": str(note.get("title") or ""),
                        "content": str(note["desc"])[:6000],
                        "url": f"https://www.xiaohongshu.com/explore/{key}",
                        "source": "小红书正文",
                    }
                )
                if getattr(self, "on_notes", None):
                    self.on_notes(notes)
        return notes

    def resolve(self, extracted, notes):
        geo = self.region()
        byid = {n["id"]: n for n in notes}
        accepted = []
        batches = []
        for p in extracted:
            name = str(p.get("name") or "").strip()
            mentions = [p.get("mention"), name, *p.get("aliases", [])]
            ids = [
                i
                for i in p.get("note_ids", [])
                if i in byid
                and any(m and str(m).casefold() in byid[i]["content"].casefold() for m in mentions)
            ]
            if not name or not ids:
                continue
            from travel_agent.reports.route import canonical_city

            city = str(p.get("city") or self.trip.destination)
            route_city = canonical_city(geo, city)
            if route_city:
                city = route_city["name"]
                p = {**p, "city": city, "city_id": route_city["id"]}
            candidates = []
            country_code = (route_city or {}).get("country_code") or geo["country_code"]
            if country_code == "CN":
                poi_candidates = []
                for lookup in list(
                    dict.fromkeys([name, p.get("mention") or name, *p.get("aliases", [])])
                )[:2]:
                    data = self.call(
                        "amap_maps__maps_text_search",
                        {"keywords": lookup, "city": city, "citylimit": True},
                    )
                    poi_candidates = main_pois(lookup, records(data, "pois"))[:2]
                    if poi_candidates:
                        break
                for candidate in poi_candidates:
                    detail = candidate
                    if not coord(detail.get("location")) or not (
                        detail.get("city") or detail.get("cityname")
                    ):
                        fetched = self.call(
                            "amap_maps__maps_search_detail", {"id": candidate["id"]}
                        )
                        detail = (records(fetched, "pois") or [fetched])[0]
                    actual_city = str(detail.get("city") or detail.get("cityname") or "")
                    if not p.get("kind") and re.search(
                        "住宿|餐饮|停车|办公", str(detail.get("type") or "")
                    ):
                        continue
                    # The provider's city, not the requested city, is authoritative.
                    expected = city.removesuffix("市")
                    actual = actual_city.removesuffix("市")
                    if (
                        coord(detail.get("location"))
                        and actual
                        and (expected in actual or actual in expected)
                    ):
                        candidates.append(
                            {
                                **detail,
                                "country_code": "CN",
                                "coordinate_system": "GCJ02",
                                "source_url": "https://uri.amap.com/marker?"
                                + urlencode(
                                    {
                                        "position": detail["location"],
                                        "name": name,
                                        "coordinate": "gaode",
                                    }
                                ),
                            }
                        )
            else:
                queries = list(dict.fromkeys([p.get("query_name") or name, *p.get("aliases", [])]))
                if len(queries) < 2:
                    queries.append(queries[0] + " " + str((route_city or {}).get("query", city)))
                for query_index, query in enumerate(queries[:2]):
                    if query_index and candidates:
                        break
                    data = self.call(
                        "open_maps__search_places",
                        {
                            "query": str(query),
                            "country_code": country_code,
                            **(
                                {"center": route_city["center"]}
                                if query_index == 0 and route_city and route_city.get("center")
                                else {}
                            ),
                        },
                    )
                    for candidate in data.get("places", []):
                        category = candidate.get("osm_value", "")
                        if not p.get("kind") and category in {
                            "cafe",
                            "restaurant",
                            "fast_food",
                            "bar",
                            "map",
                            "information",
                            "supermarket",
                        }:
                            continue
                        if category in {
                            "hotel",
                            "guest_house",
                            "hostel",
                            "parking",
                            "bus_stop",
                            "platform",
                            "primary",
                            "secondary",
                            "tertiary",
                            "residential",
                        }:
                            continue
                        if matches_base_area(route_city, {**p, "location": candidate["location"],
                                                          "actual_city": candidate.get("city")}):
                            candidates.append(candidate)
            candidates = list({str(c["id"]): c for c in candidates}.values())[:12]
            if candidates:
                batches.append(
                    {
                        "place": {**p, "name": name, "city": city, "note_ids": ids},
                        "candidates": candidates,
                    }
                )
            else:
                self.issues.append(f"地图未核验：{name}；需补查名称或位置。")
        selections = {}
        unresolved = []
        for batch in batches:
            p = batch["place"]
            aliases = {
                plain(x) for x in [p["name"], p.get("query_name", ""), *p.get("aliases", [])] if x
            }
            exact = [c for c in batch["candidates"] if plain(c.get("name", "")) in aliases]
            if len(exact) == 1:
                selections[p["name"]] = str(exact[0]["id"])
            else:
                unresolved.append(batch)
        if unresolved:
            result = self.ask(
                "从候选中选择正文对应的真实体验位置。类型必须符合kind，不能把酒店/停车场当景点。"
                "考虑城市、别名与地址，无合适候选就不选，禁止编造ID。"
                '返回 {"choices":[{"name":原始name,"id":候选id}]}。',
                {"candidates": unresolved},
                "match_places",
            )
            selections.update({c.get("name"): str(c.get("id")) for c in result.get("choices", [])})
        seen = set()
        for batch in batches:
            p = batch["place"]
            chosen = next(
                (c for c in batch["candidates"] if str(c["id"]) == selections.get(p["name"])), None
            )
            if not chosen or chosen["id"] in seen:
                continue
            seen.add(chosen["id"])
            accepted.append(
                {
                    **p,
                    "id": str(chosen["id"]),
                    "map_name": chosen.get("name"),
                    "location": chosen["location"],
                    "area": chosen.get("district") or chosen.get("adname") or p["city"],
                    "address": ", ".join(
                        str(x)
                        for x in [
                            chosen.get("address"),
                            chosen.get("street"),
                            chosen.get("district"),
                            chosen.get("city"),
                        ]
                        if x
                    ),
                    "actual_city": chosen.get("city"),
                    "country_code": chosen["country_code"],
                    "coordinate_system": chosen["coordinate_system"],
                    "source_url": byid[p["note_ids"][0]]["url"],
                    "map_url": chosen["source_url"],
                    "map_verified": True,
                }
            )
            if getattr(self, "on_points", None):
                self.on_points(accepted)
        return accepted

    def places(self, notes):
        extracted = self.ask(
            "从正文提取去重后的独立主要景点，通常准备每天3个候选；多城市均衡覆盖，最多32个。"
            "name必须逐字出现在对应正文，不能用多个景点拼接成一个。提取英文/当地语query_name和aliases便于境外地图检索。"
            "餐厅咖啡不要作为主要景点。reason只写游览价值。"
            '另总结3条有正文依据的建议或分歧。返回 {"places":[{"name":string,"city":string,"query_name":string,'
            '"aliases":[string],"reason":string,"note_ids":[string]}],"insights":[{"title":string,"text":string,"note_ids":[string]}]}。',
            {"query": self.query, "days": self.trip.duration_days, "notes": notes},
            "extract_places",
        )
        points = self.resolve(
            extracted.get("places", [])[: min(32, self.trip.duration_days * 3 + 4)], notes
        )
        ids = {n["id"] for n in notes}
        insights = [
            i
            for i in extracted.get("insights", [])
            if i.get("note_ids") and set(i["note_ids"]) <= ids
        ]
        return points, insights

    def leg(self, a, b):
        domestic = a.get("coordinate_system") == "GCJ02" and b.get("coordinate_system") == "GCJ02"
        if a.get("coordinate_system") != b.get("coordinate_system"):
            raise ValueError("Cannot mix map coordinate systems")
        tool = "amap_maps__maps_direction_driving" if domestic else "open_maps__driving_route"
        data = self.call(tool, {"origin": a["location"], "destination": b["location"]})
        access_notes = []
        if not domestic and not records(data, "paths"):
            endpoints = [a, b]
            for index, point in enumerate(endpoints):
                query = str(point.get("query_name") or point["name"]) + " entrance"
                nearby = self.call(
                    "open_maps__search_places",
                    {
                        "query": query,
                        "country_code": point["country_code"],
                        "center": point["location"],
                    },
                )
                accesses = [
                    p
                    for p in nearby.get("places", [])
                    if p.get("osm_value")
                    in {"entrance", "information", "station", "attraction", "parking"}
                    and km(p, point) < 15
                ]
                if not accesses:
                    continue
                chosen = accesses[0]
                endpoints[index] = {**point, "location": chosen["location"]}
                retried = self.call(
                    tool,
                    {"origin": endpoints[0]["location"], "destination": endpoints[1]["location"]},
                )
                if records(retried, "paths"):
                    data = retried
                    access_notes.append(
                        f"{point['name']}使用地图核验的入口/接驳点“{chosen['name']}”计算道路段；景区内部接驳或缆车时间另留。"
                    )
                    break
        paths = records(data, "paths")
        path = paths[0] if paths else {}
        seconds = number(path.get("duration")) or number((path.get("cost") or {}).get("duration"))
        distance = number(path.get("distance"))
        return {
            "from": a["name"],
            "to": b["name"],
            "mode": "驾车 / 打车接驳",
            "minutes": round(seconds / 60) if seconds else None,
            "km": round(distance / 1000, 1) if distance else None,
            "verified": bool(seconds and distance),
            "source": "高德地图" if domestic else "OSRM / OpenStreetMap（非实时拥堵）",
            "url": a["map_url"],
            "access_note": " ".join(access_notes),
        }

    def schedule(self, places, repair_context=None, attempt=0):
        n = self.trip.duration_days
        grouped = self.ask(
            "用提供的真实点位制定逐日行程，每天一般2到3个主要点，禁止重复ID。长线按城市连续停留；"
            "短途允许附近一日游，stay_city可以始终为目的地城市，避免不必要换酒店。不要按30公里机械排除周边景区。"
            "有半天/全天景区需留足时间，不能机械每2小时换景点。参考航班/高铁时刻，首末日留接驳时间。"
            '返回 {"days":[{"city":string,"stay_city":string,"area":string,"title":string,"intro":string,'
            '"stops":[{"id":真实id,"start_time":"HH:MM","duration_minutes":number}]}]}。',
            {
                "query": self.query,
                "days": n,
                "places": places,
                "transport": self.transport_options,
                "previous_failed_schedule": repair_context,
                "repair_instruction": "重新分组超时组合，全天景区搭配近距离轻量点；首末日适配交通窗口，不要在全天景区后再安排远处古镇。"
                if repair_context
                else "",
            },
            "schedule",
        )
        byid = {p["id"]: p for p in places}
        used = set()
        days = []
        for index in range(n):
            raw = (grouped.get("days", []) + [{}] * n)[index]
            stops = []
            for s in raw.get("stops", [])[:3]:
                key = str(s.get("id"))
                if key in byid and key not in used:
                    stops.append(
                        {
                            **byid[key],
                            "start_time": str(s.get("start_time") or "09:30"),
                            "duration_minutes": max(
                                30, min(480, int(s.get("duration_minutes") or 90))
                            ),
                        }
                    )
                    used.add(key)
            city = raw.get("city") or (stops[0]["city"] if stops else self.trip.destination)
            days.append(
                {
                    "day": index + 1,
                    "date": (self.trip.travel_dates.start_date + timedelta(days=index)).isoformat(),
                    "city": city,
                    "stay_city": raw.get("stay_city") or city,
                    "area": raw.get("area") or city,
                    "title": raw.get("title") or f"第{index + 1}天",
                    "intro": raw.get("intro") or "按片区游览。",
                    "stops": stops,
                    "legs": [],
                    "food": [],
                }
            )
        # Underfilled days trigger extra researched candidates, never duplicate a stop.
        for day in days:
            while len(day["stops"]) < 2:
                remaining = [
                    p
                    for p in places
                    if p["id"] not in used
                    and (
                        p["city"] == day["city"]
                        or (n <= 4 and day["stops"] and km(p, day["stops"][-1]) <= 55)
                    )
                ]
                if not remaining:
                    donor = next(
                        (
                            d
                            for d in days
                            if len(d["stops"]) > 2
                            and (
                                d["city"] == day["city"]
                                or (
                                    n <= 4
                                    and day["stops"]
                                    and any(km(s, day["stops"][0]) < 55 for s in d["stops"])
                                )
                            )
                        ),
                        None,
                    )
                    if donor:
                        moved = (
                            min(donor["stops"], key=lambda s: km(s, day["stops"][0]))
                            if day["stops"]
                            else donor["stops"][-1]
                        )
                        donor["stops"].remove(moved)
                        day["stops"].append(moved)
                        continue
                    break
                anchor = day["stops"][-1] if day["stops"] else remaining[0]
                p = min(remaining, key=lambda p: km(anchor, p))
                if day["stops"] and km(anchor, p) > 55:
                    break
                day["stops"].append({**p, "start_time": "15:00", "duration_minutes": 90})
                used.add(p["id"])
        for day in days:
            day["legs"] = [self.leg(a, b) for a, b in itertools.pairwise(day["stops"])]
            self.adjust_times(day)
            if len(day["stops"]) < 2:
                day["exception_reason"] = "受交通时刻或可核验点位限制，未凑数添加景点。"
        if attempt < 2 and any(d.get("timing_warning") or len(d["stops"]) < 2 for d in days):
            self.db.append_event(
                self.thread_id,
                self.trip.user_id,
                self.trip.trip_id,
                "report.schedule_repair",
                {
                    "attempt": attempt + 1,
                    "failed_days": [
                        d["day"] for d in days if d.get("timing_warning") or len(d["stops"]) < 2
                    ],
                },
            )
            return self.schedule(places, days, attempt + 1)
        return days

    def adjust_times(self, day):
        start = 9 * 60 + 30
        end = 21 * 60
        for t in self.transport_options:
            option = next(
                (o for o in t["options"] if o.get("baseline")), next(iter(t["options"]), {})
            )
            directions = option.get("directions", [])
            for segment in directions:
                if (
                    segment["direction"] == "outbound"
                    and segment["arrival_time"][:10] == day["date"]
                ):
                    start = max(
                        start,
                        int(segment["arrival_time"][11:13]) * 60
                        + int(segment["arrival_time"][14:16])
                        + 120,
                    )
                if (
                    segment["direction"] == "return"
                    and segment["departure_time"][:10] == day["date"]
                ):
                    end = min(
                        end,
                        int(segment["departure_time"][11:13]) * 60
                        + int(segment["departure_time"][14:16])
                        - 180,
                    )
            if t.get("type") == "高铁" and t["date"] == day["date"] and option:
                value = (
                    option.get("arrive_time")
                    if t["destination"] == day["city"]
                    else option.get("start_time")
                    if t["origin"] == day["city"]
                    else None
                )
                if value:
                    h, m = map(int, value.split(":"))
                    if t["destination"] == day["city"]:
                        start = max(start, h * 60 + m + 60)
                    else:
                        end = min(end, h * 60 + m - 60)
        road_time = sum((leg.get("minutes") or 45) + 20 for leg in day["legs"])
        activity_time = sum(stop.get("duration_minutes", 90) for stop in day["stops"])
        if start + road_time + activity_time > end and activity_time:
            ratio = (end - start - road_time) / activity_time
            if 0.65 <= ratio < 1:
                for stop in day["stops"]:
                    stop["duration_minutes"] = max(
                        30, int(stop.get("duration_minutes", 90) * ratio)
                    )
                day["intro"] += " 游览建议时长按交通窗口调整，可选择更晚返程以增加停留时间。"
        compact_schedule = any(
            re.fullmatch(r"\d{2}:\d{2}", str(s.get("start_time", "")))
            and int(s["start_time"][:2]) * 60
            + int(s["start_time"][3:])
            + sum(t.get("duration_minutes", 90) for t in day["stops"][i:])
            + sum((leg.get("minutes") or 45) + 20 for leg in day["legs"][i:])
            > end
            for i, s in enumerate(day["stops"])
        )
        day.pop("timing_warning", None)
        for index, stop in enumerate(day["stops"]):
            suggested = stop["start_time"]
            if not compact_schedule and re.fullmatch(r"\d{2}:\d{2}", suggested):
                h, m = map(int, suggested.split(":"))
                start = max(start, h * 60 + m)
            stop["start_time"] = f"{start // 60:02d}:{start % 60:02d}"
            start += stop.get("duration_minutes", 90)
            if index < len(day["legs"]):
                start += (day["legs"][index].get("minutes") or 45) + 20
        day["available_until"] = f"{end // 60:02d}:{end % 60:02d}"
        if start > end:
            day["timing_warning"] = (
                "所选交通候选与当天游览时间冲突，需要更晚返程或删减活动；当前方案尚未通过时间验收。"
            )

    def transport(self, days):
        rail_requested = bool(re.search("高铁|铁路|火车", self.query))
        if not rail_requested and self.previous.get("travel_mode") == "rail":
            rail_requested = True
        neg = bool(re.search(r"(?:不用|不查|无需|不需要).{0,6}(?:航班|机票)", self.query))
        options = []
        if rail_requested:
            city_days = [days[0]] + [
                b for a, b in itertools.pairwise(days) if a["city"] != b["city"]
            ]
            origin = self.trip.origin
            for day in city_days:
                if origin != day["city"]:
                    options.append(self.rail(origin, day["city"], day["date"]))
                origin = day["city"]
            if origin != self.trip.origin:
                options.append(
                    self.rail(origin, self.trip.origin, self.trip.travel_dates.end_date.isoformat())
                )
        elif self.trip.origin and not neg and self.trip.origin != days[0]["city"]:
            codes = self.ask(
                '确定本次往返机场IATA代码，只返回 {"origin":string,"destination":string}。',
                {
                    "origin": self.trip.origin,
                    "destination": self.geo.get("cities", [{"query": self.trip.destination}])[0][
                        "query"
                    ],
                },
                "airports",
            )
            if not all(
                re.fullmatch("[A-Z]{3}", str(codes.get(k, ""))) for k in ("origin", "destination")
            ):
                self.issues.append("无法核验机场代码，航班未查询。")
                return options
            args = {
                "origin": codes["origin"],
                "destination": codes["destination"],
                "departure_date": self.trip.travel_dates.start_date.isoformat(),
                "return_date": self.trip.travel_dates.end_date.isoformat(),
                "passengers": 1,
                "currency": self.trip.budget.currency,
                "show_all_results": True,
            }
            from travel_agent.graph.agent_nodes import _flight_argument_overrides

            flight_spec = self.tools.full_spec("google_flights__search_flights") or {}
            args.update(_flight_argument_overrides(self.trip, flight_spec.get("input_schema", {})))
            if "top_n" in flight_spec.get("input_schema", {}).get("properties", {}):
                args["top_n"] = 3
            provided_candidates = [r for r in getattr(self, "provided_results", [])
                if r.get("tool_name") == "google_flights__search_flights"
                and r.get("result", {}).get("success")
                and all(r.get("arguments", {}).get(k) == args.get(k) for k in
                        ("origin", "destination", "departure_date", "return_date", "currency"))
                and r.get("arguments", {}).get("passengers", 1) in {1, self.trip.traveler_count or 1}
                and not any(r.get("arguments", {}).get(k) for k in ("children", "infants_in_seat", "infants_on_lap"))
                and all(r.get("arguments", {}).get(k, "ECONOMY" if k == "cabin_class" else None) == v
                        for k, v in args.items() if k in {"cabin_class", "max_stops"})]
            selected_ids = {s["result_id"] for s in getattr(self, "selected_results", [])}
            provided_candidates.sort(key=lambda r: r["result"].get("result_id", r.get("execution_id")) not in selected_ids)
            provided = next(iter(provided_candidates), None)
            quote_people = int(provided["arguments"].get("passengers", 1)) if provided else 1
            quoted_at = provided["result"]["fetched_at"] if provided else None
            data = provided["result"]["data"] if provided else self.call("google_flights__search_flights", args)
            if not provided and self.results and self.results[-1].get("tool_name") == "google_flights__search_flights":
                quoted_at = self.results[-1].get("fetched_at")
            if provided:
                selection = next((s for s in getattr(self, "selected_results", [])
                                  if s["result_id"] == provided["result"].get("result_id", provided.get("execution_id"))), None)
                if selection and selection["item_index"] < len(data.get("flights", [])):
                    data = {**data, "flights": [data["flights"][selection["item_index"]]]}
            if not data.get("flights"):
                quote_people, quoted_at = 1, None
                data = self.call(
                    "google_flights__search_flights", {**args, "show_all_results": False}
                )
                if self.results and self.results[-1].get("tool_name") == "google_flights__search_flights":
                    quoted_at = self.results[-1].get("fetched_at")
            flights = []
            for flight in data.get("flights", [])[:20]:
                legs = flight.get("legs", [])
                if len(legs) < 2 or number(flight.get("price")) is None:
                    continue
                # fli v0.9 flattens outbound/return legs into one list. Split
                # at the actual destination, not at arbitrary dates or halfway.
                dest_names = {
                    plain(legs[0].get("departure_airport_name")),
                    plain(legs[0].get("departure_airport")),
                }
                split = next(
                    (
                        i + 1
                        for i, leg in enumerate(legs[:-1])
                        if plain(leg.get("arrival_airport_name"))
                        == plain(legs[i + 1].get("departure_airport_name"))
                        and str(legs[i + 1].get("departure_time", ""))[:10] == args["return_date"]
                    ),
                    None,
                )
                if split is None:
                    continue
                directions = []
                for direction, group in [("outbound", legs[:split]), ("return", legs[split:])]:
                    first, last = group[0], group[-1]
                    if (
                        direction == "outbound"
                        and plain(last.get("arrival_airport_name")) in dest_names
                    ):
                        continue
                    directions.append(
                        {
                            "direction": direction,
                            "origin": first.get("departure_airport_name")
                            or first.get("departure_airport"),
                            "destination": last.get("arrival_airport_name")
                            or last.get("arrival_airport"),
                            "departure_time": first["departure_time"],
                            "arrival_time": last["arrival_time"],
                            "flight_numbers": " / ".join(
                                str(leg.get("airline_code", "")) + str(leg.get("flight_number", ""))
                                for leg in group
                            ),
                            "duration_minutes": sum(leg.get("duration", 0) for leg in group),
                            "stops": len(group) - 1,
                            "legs": group,
                        }
                    )
                if len(directions) != 2 or any(d["origin"] == d["destination"] for d in directions):
                    continue
                source_price = number(flight["price"])
                price = round(source_price / quote_people, 2)
                group_quoted = quote_people == (self.trip.traveler_count or 1)
                flights.append(
                    {
                        "directions": directions,
                        "price": price,
                        "currency": flight.get("currency", self.trip.budget.currency),
                        "per_person_price": price,
                        "group_estimated_price": round(price * (self.trip.traveler_count or 1), 2),
                        "source_price": source_price,
                        "source_price_unit": "total_for_search_party",
                        "source_result_id": provided["result"].get("result_id", provided.get("execution_id")) if provided else None,
                        "group_price_is_quote": group_quoted,
                        "quote_passengers": quote_people,
                        "requested_passengers": self.trip.traveler_count or 1,
                        "price_basis": f"每人往返平均参考价（{quote_people}人合计报价÷{quote_people}）",
                        "group_price_basis": "本次查询人数的往返合计报价，库存未锁定" if group_quoted else "按单人价格乘人数估算，全部乘客同价与库存需复核",
                        "baggage": "行李额度及附加费用需在航空公司确认",
                        "cabin_class": args.get("cabin_class", "ECONOMY"),
                        "comfort_preference": bool(
                            self.trip.preferences.get("flight", {}).get("comfort")
                        ),
                        "fetched_at": quoted_at or datetime.now(UTC).isoformat(),
                        "route": f"{codes['origin']} ↔ {codes['destination']}",
                    }
                )
            # Prefer a late return at the same/lower cost, to make the final day usable.
            comfort = bool(self.trip.preferences.get("flight", {}).get("comfort"))
            flights.sort(
                key=lambda f: (
                    f["directions"][1]["arrival_time"][:10]
                    > self.trip.travel_dates.end_date.isoformat(),
                    sum(d.get("stops", 0) for d in f["directions"]) if comfort else 0,
                    sum(d.get("duration_minutes", 0) for d in f["directions"]) if comfort else 0,
                    f["directions"][0]["arrival_time"][:10] != args["departure_date"],
                    f["directions"][0]["arrival_time"][11:16] > "13:00",
                    f["directions"][1]["departure_time"][11:16] < "16:00",
                    f["price"],
                    f["directions"][0]["arrival_time"],
                )
            )
            for i, flight in enumerate(flights[:3]):
                flight["baseline"] = i == 0
            options.append(
                {
                    "type": "往返航班",
                    "origin": self.trip.origin,
                    "destination": self.trip.destination,
                    "date": args["departure_date"],
                    "options": flights[:3],
                    "url": "https://www.google.com/travel/flights?"
                    + urlencode(
                        {
                            "q": f"Flights from {codes['origin']} to {codes['destination']} on {args['departure_date']} through {args['return_date']}",
                            "curr": self.trip.budget.currency,
                        }
                    ),
                    "fetched_at": quoted_at or datetime.now(UTC).isoformat(),
                    "raw_summary": f"已采用{quote_people}人往返合计报价，折算每人参考价；组合报价不拆成虚构单程价格。"
                    if flights
                    else "真实航班查询未返回可用组合，预算使用明确预留。",
                }
            )
        for item in options:
            item.setdefault("fetched_at", datetime.now(UTC).isoformat())
        return sorted(options, key=lambda t: t["date"])

    def hotels(self, days):
        groups = []
        from pycountry import countries

        from travel_agent.reports.route import external_windows

        arrival, departure = external_windows(self.transport_options)
        for day in days[:-1]:
            if day.get("no_hotel"):
                continue
            if arrival.get("arrival_time") and day["date"] < arrival["arrival_time"][:10]:
                continue
            if departure.get("departure_time") and day["date"] >= departure["departure_time"][:10]:
                continue
            city = day.get("stay_city") or day["city"]
            if not groups or groups[-1]["city"] != city or groups[-1]["end_day"] + 1 != day["day"]:
                groups.append(
                    {
                        "city": city,
                        "area": day["area"],
                        "start_day": day["day"],
                        "end_day": day["day"],
                        "hotels": [],
                    }
                )
            groups[-1]["end_day"] = day["day"]
        rooms = math.ceil((self.trip.traveler_count or 2) / 2)
        for g in groups:
            from travel_agent.reports.route import canonical_city
            city_identity = canonical_city(self.geo, g["city"]) or {}
            country_code = city_identity.get("country_code") or self.geo["country_code"]
            country = countries.get(alpha_2=country_code)
            country_name = country.name if country else self.geo["country_name"]
            start = self.trip.travel_dates.start_date
            checkin = start + timedelta(days=g["start_day"] - 1)
            checkout = start + timedelta(days=min(g["end_day"], len(days) - 1))
            if checkout <= checkin:
                continue
            g.update(
                checkin=checkin.isoformat(),
                checkout=checkout.isoformat(),
                nights=(checkout - checkin).days,
            )
            if self.hotel_targets and g["city"] not in self.hotel_targets:
                retained = next(
                    (
                        old
                        for old in self.previous.get("hotel_regions", [])
                        if old["city"] == g["city"]
                        and old.get("checkin") == g["checkin"]
                        and old.get("checkout") == g["checkout"]
                        and old.get("hotels")
                    ),
                    None,
                )
                if retained:
                    g.clear()
                    g.update(copy.deepcopy(retained))
                continue
            query_args = {
                    "destination": next(
                        (
                            c["query"]
                            for c in self.geo["cities"]
                            if c["name"].removesuffix("市") == g["city"].removesuffix("市")
                        ),
                        g["city"],
                    )
                    + ", "
                    + country_name,
                    "checkIn": g["checkin"],
                    "checkOut": g["checkout"],
                    "adults": self.trip.traveler_count or 2,
                    "rooms": rooms,
                    "maxResults": 8,
                }
            aliases = [g["city"], city_identity.get("query", ""), *city_identity.get("aliases", [])]
            provided = next((r for r in getattr(self, "provided_results", [])
                if r.get("tool_name") == "booking__booking_search"
                and r.get("result", {}).get("success")
                and all(r.get("arguments", {}).get(k) == query_args.get(k) for k in ("checkIn", "checkOut", "adults", "rooms"))
                and any(a and a.casefold() in str(r.get("arguments", {}).get("destination", "")).casefold() for a in aliases)), None)
            data = provided["result"]["data"] if provided else self.call("booking__booking_search", query_args)
            if not data.get("properties", data.get("hotels", [])):
                from travel_agent.reports.route import canonical_city

                target_city = canonical_city(self.geo, g["city"])
                alternate_query = (target_city or {}).get("query") or g["city"]
                data = self.call(
                    "booking__booking_search",
                    {
                        "destination": alternate_query,
                        "checkIn": g["checkin"],
                        "checkOut": g["checkout"],
                        "adults": self.trip.traveler_count or 2,
                        "rooms": rooms,
                        "maxResults": 8,
                    },
                )
            seen = set()
            candidates = []
            choices = getattr(self, "hotel_choices", {})
            selected_url = choices.get(g["city"])
            properties = list(data.get("properties", data.get("hotels", [])))
            if selected_url and not any(
                urlsplit(p.get("source_url") or p.get("url") or "").path == urlsplit(selected_url).path
                for p in properties
            ):
                for execution in reversed(self.db.list_tool_executions(self.thread_id, limit=1000)):
                    result = execution["payload"]["result"]
                    if execution["tool_name"] != "booking__booking_search" or not result["success"]:
                        continue
                    previous_data = result.get("data", {})
                    found = next((
                        p for p in previous_data.get("properties", previous_data.get("hotels", []))
                        if urlsplit(p.get("source_url") or p.get("url") or "").path == urlsplit(selected_url).path
                    ), None)
                    if found:
                        properties.insert(0, found)
                        break
            for p in properties:
                if re.search(
                    r"\b(?:cruise|balloon|excursion|guided tour)\b", str(p.get("name", "")), re.I
                ) and not re.search("邮轮|游轮|热气球住宿", self.query):
                    continue
                url = p.get("source_url") or p.get("url")
                if (
                    not isinstance(url, str)
                    or not url.startswith("https://")
                    or urlsplit(url).path in seen
                ):
                    continue
                if not urlsplit(url).path.startswith(
                    "/hotel/" + country_code.lower() + "/"
                ):
                    self.issues.append(f"排除国家不符的住宿：{p.get('name', '未知酒店')}。")
                    continue
                seen.add(urlsplit(url).path)
                candidates.append(p)
            prior_names = {
                h["name"] for old in self.previous.get("hotel_regions", []) for h in old["hotels"]
            }
            if re.search("换|更换|重新选", self.query):
                candidates.sort(
                    key=lambda p: (
                        p.get("name") in prior_names,
                        -float(p.get("reviewScore") or p.get("rating") or 0),
                    )
                )
            if selected_url:
                selected_path = urlsplit(selected_url).path
                if not any(urlsplit(p.get("source_url") or p.get("url") or "").path == selected_path for p in candidates):
                    raise ValueError("指定酒店不在本旅行的真实检索记录中，先搜索并提供来源中的URL；不能伪造房源")
                candidates.sort(key=lambda p: urlsplit(p.get("source_url") or p.get("url") or "").path != selected_path)
            for index, p in enumerate(candidates[:2]):
                url = p.get("source_url") or p["url"]
                evidence = self.call(
                    "booking_evidence__read_quote",
                    {
                        "url": url,
                        "name": p["name"],
                        "destination": next(
                            (
                                c["query"]
                                for c in self.geo["cities"]
                                if c["name"].removesuffix("市") == g["city"].removesuffix("市")
                            ),
                            g["city"],
                        )
                        + ", "
                        + country_name,
                        "checkin": g["checkin"],
                        "checkout": g["checkout"],
                        "adults": self.trip.traveler_count or 2,
                        "rooms": rooms,
                        "currency": self.trip.budget.currency,
                    },
                )
                quote_fetched_at = self.results[-1].get("fetched_at") if self.results and self.results[-1].get("tool_name") == "booking_evidence__read_quote" else None
                texts = (
                    [c.get("text", "") for c in evidence.get("cards", [])]
                    + [r.get("text", "") for r in evidence.get("rows", [])]
                    + evidence.get("price_context", [])
                )
                parsed = visible_hotel_quote(
                    evidence,
                    g["nights"],
                    self.trip.traveler_count or 2,
                    rooms,
                    self.trip.budget.currency,
                    g["checkin"],
                    g["checkout"],
                )
                parsed = parsed or {
                    "price_is_total": False,
                    "total_price": None,
                    "currency": "",
                    "price_note": "未取得明示计价范围的报价",
                }
                excerpt = str(parsed.get("evidence_excerpt") or "")
                verified = bool(
                    parsed.get("price_is_total")
                    and number(parsed.get("total_price")) is not None
                    and excerpt
                    and any(compact(excerpt) in compact(t) for t in texts)
                    and re.fullmatch("[A-Z]{3}", str(parsed.get("currency", "")))
                )
                if verified:
                    visible_amount = f"{number(parsed['total_price']):g}"
                    verified = bool(
                        re.search(
                            r"(?<!\d)" + re.escape(visible_amount) + r"(?!\d)",
                            " ".join(texts).replace(",", ""),
                        )
                    )
                total = number(parsed.get("total_price")) if verified else None
                currency = (
                    parsed.get("currency") if verified else str(p.get("currency") or "币种未明")
                )
                g["hotels"].append(
                    {
                        "name": p["name"],
                        "url": url,
                        "image_url": p.get("imageUrl")
                        or next(iter(evidence.get("images", [])), ""),
                        "images": list(
                            dict.fromkeys(
                                [
                                    u
                                    for u in [p.get("imageUrl"), *evidence.get("images", [])]
                                    if isinstance(u, str) and u.startswith("https://")
                                ]
                            )
                        )[:8],
                        "total_price": total,
                        "nightly_price": round(total / g["nights"] / rooms, 2)
                        if total is not None
                        else None,
                        "currency": currency,
                        "price_is_total": verified,
                        "quote_version": 2,
                        "price_note": (parsed.get("price_note") or "已核验入住报价")
                        if verified
                        else "未取得币种与计价周期完整的报价，预算使用规划预留。",
                        "display_price": str(p.get("priceNote") or ""),
                        "quote_excerpt": excerpt if verified else "",
                        "tax_note": parsed.get("tax_note") or "税费范围需复核",
                        "room_name": parsed.get("room_name") or "房型待核实",
                        "rating": p.get("reviewScore") or p.get("rating"),
                        "location": p.get("location") or g["area"],
                        "rooms": rooms,
                        "adults": self.trip.traveler_count or 2,
                        "baseline": index == 0,
                        "fetched_at": quote_fetched_at or datetime.now(UTC).isoformat(),
                        "pros": [
                            f"平台评分 {p.get('reviewScore') or p.get('rating')}"
                            if p.get("reviewScore") or p.get("rating")
                            else "匹配本次住宿城市",
                            "搜索卡片标注含早餐"
                            if p.get("breakfastIncluded")
                            else "可在详情页查看配套设施",
                        ],
                        "cons": [
                            "退改条款和具体房型需核实",
                            parsed.get("tax_note") or "税费范围需核实",
                        ],
                    }
                )
                if getattr(self, "on_hotels", None):
                    self.on_hotels(groups)
        return groups

    def food(self, notes, days):
        result = self.ask(
            "从真实笔记正文提取具体餐厅、小吃店和咖啡馆，最多10家。name必须逐字出现在引用正文；不造店名和人均价格。"
            '菜品、人均/营业信息只保留正文有依据的内容并注明经验参考。返回 {"food":[{"name":string,"city":string,'
            '"query_name":string,"aliases":[string],"kind":"餐厅|小吃|咖啡","reason":string,"recommended_items":[string],'
            '"price_reference":string,"note_ids":[string]}]}。',
            {"notes": notes, "query": self.query},
            "food_extract",
        )
        raw = result.get("food", [])
        verified = self.resolve(raw, notes)
        ids = {p["name"]: p for p in verified}
        byid = {n["id"]: n for n in notes}
        food = []
        for p in raw:
            name = str(p.get("name") or "")
            note_ids = [
                i for i in p.get("note_ids", []) if i in byid and name in byid[i]["content"]
            ]
            if not note_ids:
                continue
            item = {
                **p,
                **ids.get(name, {}),
                "note_ids": note_ids,
                "source_url": byid[note_ids[0]]["url"],
                "map_verified": name in ids,
            }
            if name not in ids:
                item["map_url"] = "https://www.openstreetmap.org/search?" + urlencode(
                    {"query": name + " " + p.get("city", self.trip.destination)}
                )
                item["location_status"] = "正文推荐，具体位置待核验"
            food.append(item)
        return food

    def assign_food(self, food, days):
        used = set()
        for day in days:
            candidates = [
                f
                for f in food
                if f["name"] not in used and (f.get("city") == day["city"] or len(days) <= 4)
            ]
            if day["stops"]:

                def distance(f):
                    return min(km(f, s) for s in day["stops"]) if coord(f.get("location")) else 1000

                candidates.sort(key=distance)
            day["food"] = []
            for f in candidates[:2]:
                if (
                    day["stops"]
                    and coord(f.get("location"))
                    and min(km(f, s) for s in day["stops"]) > 15
                ):
                    continue
                day["food"].append(f["name"])
                used.add(f["name"])

    def run(self):
        from travel_agent.reports.orchestration import run_adaptive

        return run_adaptive(self)
