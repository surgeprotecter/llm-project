"""Evidence-backed one-way transport; no workflow or budget coupling.

plan_intercity(planner, stays) returns adjacent stay transfers, in order.
plan_entry_exit researches only the two boundary edges. MobilityPlanner is an
optional mixin; external functions work directly with StandardTripPlanner.

Stays accept id/city_id, name/city, query, country_code, center, start_day,
end_day, checkin, checkout. geo.cities supplies missing fields by identity.
Incoming transfers use checkin; return uses last checkout. Times are local.
Unknowns remain None. Group/vehicle fares are never silently divided.

Every edge and option has status quote|reference|estimate|unavailable. Options
have baseline, price_low/high, currency, price_basis, dates, times, duration,
availability and evidence. Consumers must support these price semantics before
budget aggregation. Prior report options are not reused as fresh quotes.

travel_policy on planner or geo accepts entry_mode, intercity_mode, exit_mode
and edges={edge_id: mode}. geo.entry_mode/exit_mode are also supported.
One batched model call chooses all modes and airport codes, then at
most one call extracts web evidence. No model-only fares are used.

Limits per invocation: 18 tool calls, 6 web searches, one provider lookup
per edge, 3 options per edge. Registry retries are outside these limits.
A failed web service is disabled for remaining edges. No direct HTTP calls.
"""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit

from jsonschema import exceptions, validators

from travel_agent.reports.planning import records
from travel_agent.tools.normalization import flight_elapsed_minutes

__all__ = ["MobilityPlanner", "plan_intercity", "plan_entry_exit"]

_LABELS = {"rail": "铁路", "bus": "城际巴士", "flight": "航班", "ferry": "渡轮", "car": "公路估算"}
_MODES = {"rail", "bus", "flight", "ferry", "car"}
_ALIASES = {"train": "rail", "air": "flight", "road": "bus"}
_FLIGHTS = "google_flights__search_flights"
_RAIL = "china_rail__get-tickets"
_ROAD = "open_maps__driving_route"
_NUMBER = r"\d+(?:[.,]\d{3})*(?:[.,]\d+)?"
_MONEY = re.compile(
    rf"(?P<before>\b[A-Z]{{3}}\b|RMB|US\$|HK\$|S\$|A\$|C\$|人民币|美元|港币|越南盾|[¥￥₫€£$])?"
    rf"\s*(?P<amount>{_NUMBER})\s*(?P<scale>万|千|[kKwW](?![a-zA-Z]))?\s*"
    r"(?P<after>\b[A-Z]{3}\b|RMB|人民币|美元|港币|越南盾|[元盾₫€£])?"
)


def _num(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def _day(value):
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError:
        return None


def _mode(value):
    value = str(value or "").lower()
    return _ALIASES.get(value, value)

_RAIL_COUNTRIES = {"CN", "HK", "MO"}
_COUNTRY_ALIASES = {
    "香港": "HK",
    "香港特别行政区": "HK",
    "hong kong": "HK",
    "hongkong": "HK",
    "hk": "HK",
    "澳门": "MO",
    "澳门特别行政区": "MO",
    "macau": "MO",
    "macao": "MO",
    "mo": "MO",
    "中国": "CN",
    "中国大陆": "CN",
    "内地": "CN",
    "china": "CN",
    "mainland china": "CN",
}


def country_code(name):
    """Best-effort ISO alpha-2 code for an origin/city name."""
    text = str(name or "").strip()
    if not text:
        return ""
    if text in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[text]
    key = text.lower()
    if key in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[key]
    try:
        import pycountry

        try:
            return pycountry.countries.lookup(text).alpha_2
        except LookupError:
            pass
        matches = pycountry.countries.search_fuzzy(text)
        return matches[0].alpha_2 if matches else ""
    except (ImportError, LookupError, AttributeError, IndexError):
        return ""


def city_matches(value, city):
    left, right = str(value or "").strip(), str(city or "").strip()
    if not left or not right:
        return False
    return left in right or right in left



def _clock(value):
    match = re.search(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)", str(value or ""))
    if match and int(match[1]) < 24:
        return f"{int(match[1]):02}:{match[2]}"
    return None


def _duration(value):
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+):([0-5]\d)\s*", value)
        if match:
            return int(match[1]) * 60 + int(match[2])
        match = re.fullmatch(r"\s*(?:(\d+)\s*(?:h|小时))?\s*(?:(\d+)\s*(?:m|min|分钟))?\s*", value)
        if match and any(match.groups()):
            return int(match[1] or 0) * 60 + int(match[2] or 0)
    return _num(value)


def _times(when, departure=None, arrival=None, duration=None, arrival_date=None, flight=False):
    start_day = _day(departure) or _day(when)
    end_day = _day(arrival) or _day(arrival_date)
    start_time, end_time = _clock(departure), _clock(arrival)
    minutes = _duration(duration)
    start = (
        datetime.fromisoformat(f"{start_day}T{start_time}") if start_day and start_time else None
    )
    if not end_day and start:
        if minutes is not None and not flight:
            end_day = (start + timedelta(minutes=minutes)).date().isoformat()
        elif end_time:
            end_day = (start + timedelta(days=end_time < start_time)).date().isoformat()
    end = datetime.fromisoformat(f"{end_day}T{end_time}") if end_day and end_time else None
    if minutes is None and start and end and end >= start and not flight:
        minutes = (end - start).total_seconds() / 60
    return {
        "departure_date": start_day,
        "departure_time": start_time,
        "arrival_date": end_day,
        "arrival_time": end_time,
        "duration_minutes": minutes,
    }


def _city(stay, cities):
    value = dict(stay) if isinstance(stay, dict) else {"city": str(stay or "")}
    identity = value.get("city_id") or value.get("id")
    name = value.get("city") or value.get("name") or value.get("query")
    canonical = next(
        (c for c in cities if identity and identity in (c.get("id"), c.get("city_id"))), None
    )
    if canonical is None and name:
        canonical = next(
            (
                c
                for c in cities
                if name in [c.get("name"), c.get("city"), c.get("query"), *c.get("aliases", [])]
            ),
            None,
        )
    value = {**(canonical or {}), **value}
    value["city"] = value.get("city") or value.get("name") or value.get("query") or ""
    value["name"] = value["city"]
    value["query"] = value.get("query") or value["city"]
    value["city_id"] = value.get("city_id") or value.get("id")
    value["id"] = value.get("id") or value["city_id"]
    return value


def _edges(planner, stays, boundary_only):
    geo = getattr(planner, "geo", {}) or {}
    cities = geo.get("cities", [])
    stays = [_city(s, cities) for s in stays]
    if not stays:
        return []
    dates = planner.trip.travel_dates
    start, finish = _day(dates.start_date), _day(dates.end_date)
    for stay in stays:
        if not _day(stay.get("checkin")) and start and _num(stay.get("start_day")):
            stay["checkin"] = (
                date.fromisoformat(start) + timedelta(days=int(stay["start_day"]) - 1)
            ).isoformat()
    origin = _city(getattr(planner.trip, "origin", None), cities)
    origin_country = (
        str(origin.get("country_code") or "").upper()
        or str(geo.get("origin_country_code") or "").upper()
        or str(country_code(origin.get("city")) or "").upper()
    )
    if not origin_country and str(geo.get("entry_mode") or "").lower() in {"", "rail", "bus", "car"}:
        # Same-country land routes inherit the destination country: mainland city names are not
        # in pycountry, so 北京→上海 would otherwise lose its country code and skip 12306.
        origin_country = str(geo.get("country_code") or "").upper()
    origin["country_code"] = origin_country
    chain, result = [origin, *stays, origin], []
    for index, (source, target) in enumerate(zip(chain, chain[1:])):
        kind = "entry" if index == 0 else "exit" if index == len(chain) - 2 else "intercity"
        if (boundary_only and kind == "intercity") or (not boundary_only and kind != "intercity"):
            continue
        when = (
            _day(target.get("checkin"))
            if kind != "exit"
            else _day(source.get("checkout")) or finish
        )
        edge = {
            "id": f"edge:{index}:{source.get('city_id') or source['city']}:{target.get('city_id') or target['city']}",
            "kind": kind,
            "scope": "intercity" if kind == "intercity" else "external",
            "origin": source["city"],
            "destination": target["city"],
            "origin_id": source.get("city_id"),
            "destination_id": target.get("city_id"),
            "from": source,
            "to": target,
            "date": when,
            "status": "unavailable",
            "options": [],
            "gaps": [],
            "url": "",
        }
        if not when:
            edge["gaps"].append("Transfer date missing; dated availability was not queried.")
        result.append(edge)
    return result


def _option(edge, mode, status, evidence, **values):
    item = {
        "mode": mode,
        "status": status,
        "baseline": False,
        "price": None,
        "price_low": None,
        "price_high": None,
        "currency": None,
        "price_basis": "unknown",
        "people": None,
        "availability": "unknown",
        "bookable": False,
        "source_url": evidence.get("url") or "",
        "evidence": evidence,
        **_times(edge["date"]),
    }
    item.update(values)
    item["price_unit"] = (
        "group" if item["price_basis"] in {"per_group", "per_vehicle"} else "per_person"
    )
    if item["price_low"] is None:
        item["price_low"] = item["price"]
    if item["price_high"] is None:
        item["price_high"] = item["price"]
    item["type"] = _LABELS[mode]
    item["name"] = item.get("name") or item["type"]
    item["route"] = f"{edge['origin']} → {edge['destination']} · {item['type']} · {item['name']}"
    minutes = item.get("duration_minutes")
    item["durationText"] = f"约 {minutes:g} 分钟" if minutes is not None else "耗时待核实"
    item["url"] = item["source_url"]
    item["price_basis_text"] = {
        "per_person": "每人单程",
        "per_group": "全团单程",
        "per_vehicle": "每车单程",
        "unknown": "计价单位待核实",
    }.get(item["price_basis"], "计价单位待核实")
    return item


class _Research:
    def __init__(self, planner, edges):
        self.planner, self.edges = planner, edges
        self.calls = self.searches = 0
        self.web_failed = False
        self.cache, self.sources = {}, []
        self.saved_flights = {}
        for record in getattr(planner, "provided_results", []):
            args = record.get("arguments", {})
            result = record.get("result", {})
            data = result.get("data", {})
            if (record.get("tool_name") == _FLIGHTS and result.get("success") and data.get("flights")
                and result.get("source_type") in {"mcp", "api", "cli"}
                and str(data.get("trip_type")).lower() == "one_way" and not args.get("return_date")
                and args.get("passengers", 1) == 1 and result.get("result_id")):
                self.saved_flights[result["result_id"]] = record

    def saved_for_edge(self, edge):
        return {
            ref: record for ref, record in self.saved_flights.items()
            if record["arguments"].get("departure_date") == edge["date"]
        }

    def gap(self, edge, message):
        if message not in edge["gaps"]:
            edge["gaps"].append(message)

    def spec(self, name):
        try:
            value = self.planner.tools.full_spec(name)
            return value if isinstance(value, dict) else None
        except (AttributeError, KeyError):
            return None

    def call(self, edge, name, values):
        spec = self.spec(name)
        if not spec:
            self.gap(edge, f"Tool not connected: {name}")
            return {}, {}
        schema = spec.get("input_schema") or {}
        properties = schema.get("properties", {})
        args = {k: v for k, v in values.items() if k in properties and v is not None}
        for key, prop in properties.items():
            if key not in args and key in schema.get("required", []) and "default" in prop:
                args[key] = prop["default"]
        try:
            validators.validator_for(schema)(schema).validate(args)
        except (exceptions.ValidationError, exceptions.SchemaError):
            self.gap(edge, f"Cannot satisfy discovered schema: {name}")
            return {}, {}
        key = name, json.dumps(args, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        if self.calls >= 18:
            self.gap(edge, "Transport tool-call limit reached.")
            return {}, {}
        self.calls += 1
        previous = len(getattr(self.planner, "results", []))
        try:
            data = self.planner.call(name, args)
        except Exception as exc:  # optional external capability boundary
            self.gap(edge, f"{name} failed ({type(exc).__name__}).")
            if name == "web_search":
                self.web_failed = True
            return {}, {}
        results = getattr(self.planner, "results", [])
        meta = results[-1] if len(results) > previous else {}
        failed = meta.get("success") is False or (
            isinstance(data, dict) and data.get("success") is False
        )
        if failed or meta.get("source_type") == "mock":
            self.gap(edge, f"{name}: no usable live evidence.")
            if name == "web_search":
                self.web_failed = True
            data = {}
        evidence = {
            "tool": name,
            "url": meta.get("source_url") or "",
            "fetched_at": meta.get("fetched_at"),
            "validation": "provider_response",
        }
        value = (data if isinstance(data, dict) else {}, evidence)
        self.cache[key] = value
        return value

    def choose(self):
        geo = getattr(self.planner, "geo", {})
        policy = getattr(self.planner, "travel_policy", None) or geo.get("travel_policy", {})
        if not isinstance(policy, dict):
            policy = {}
        inputs = [
            {
                **{k: e[k] for k in ("id", "kind", "date", "from", "to")},
                "previous_mode": e.get("mode"),
                "origin_airport": e.get("origin_airport"),
                "destination_airport": e.get("destination_airport"),
                "saved_flights": [
                    {"ref": ref, **{k: record["arguments"].get(k) for k in ("origin", "destination")},
                     "from_airport": legs[0].get("departure_airport_name"),
                     "to_airport": legs[-1].get("arrival_airport_name")}
                    for ref, record in self.saved_for_edge(e).items()
                    if (legs := record["result"]["data"]["flights"][0].get("legs"))
                ],
            }
            for e in self.edges
        ]
        try:
            chosen = (
                self.planner.ask(
                    "Choose plausible intercity modes ONCE for all edges. No fares or invented facts. Respect the current user query: an explicit flight requery must remain flight; never replace it with a bus. Keep previous_mode unless the user asks to change mode. "
                    "Respect travel_policy. CN rail requires two CN endpoints. road means bus research; "
                    "Use saved_flights refs matching both endpoints first. If the origin or destination has approved alternatives, "
                    "select the saved refs for all matching alternatives. Airport code fields must be single IATA codes. "
                    "For a town without an airport, use a confirmed nearby commercial gateway and retain the ground connection. "
                    "car is a road estimate, not bus availability. Return JSON "
                    '{"edges":[{"id":string,"mode":"rail|bus|flight|ferry|car",'
                    '"origin_airport":string|null,"destination_airport":string|null,"flight_refs":[string]}]}.',
                    {"query": self.planner.query, "edges": inputs, "travel_policy": policy},
                    "mobility_modes",
                )
                or {}
            )
        except Exception:
            chosen = {}
        choices = chosen.get("edges", []) if isinstance(chosen, dict) else []
        selected = (
            {
                v.get("id"): v
                for v in choices
                if isinstance(v, dict) and isinstance(v.get("id"), str)
            }
            if isinstance(choices, list)
            else {}
        )
        for edge in self.edges:
            item = selected.get(edge["id"], {})
            key = "intercity_mode" if edge["kind"] == "intercity" else edge["kind"] + "_mode"
            overrides = policy.get("edges") if isinstance(policy.get("edges"), dict) else {}
            explicit = overrides.get(edge["id"]) or policy.get(key) or geo.get(key)
            mode = _mode(explicit or item.get("mode"))
            edge["mode"] = mode if mode in _MODES else self.default_mode(edge)
            edge["type"] = _LABELS[edge["mode"]]
            available = self.saved_for_edge(edge)
            refs = item.get("flight_refs")
            if isinstance(refs, list):
                edge["flight_refs"] = list(dict.fromkeys(ref for ref in refs if isinstance(ref, str) and ref in available))
            for key in ("origin_airport", "destination_airport"):
                candidate = item.get(key)
                if isinstance(candidate, str) and re.fullmatch(r"[A-Z]{3}", candidate):
                    edge[key] = candidate

    @staticmethod
    def rail_capable(edge):
        """12306 covers mainland China plus Hong Kong West Kowloon/Macau connections."""
        countries = {str(edge[v].get("country_code", "")).upper() for v in ("from", "to")}
        return bool(countries) and countries <= _RAIL_COUNTRIES and "CN" in countries

    @staticmethod
    def domestic(edge):
        return MobilityPlanner.rail_capable(edge)

    def default_mode(self, edge):
        if self.rail_capable(edge):
            return "rail"
        return "flight" if edge["kind"] in {"entry", "exit", "intercity"} else "bus"

    def provider(self, edge):
        if not edge["date"] or not edge["origin"] or not edge["destination"]:
            return
        mode = edge.get("mode") or self.default_mode(edge)
        edge["mode"] = mode
        if mode == "rail":
            if not self.rail_capable(edge):
                self.gap(
                    edge,
                    "铁路查询只覆盖中国内地与香港/澳门之间的线路；当前出发地国家码未知或该线路未开通铁路。",
                )
                return
            if not self.rail(edge):
                self.gap(edge, "12306 未返回可售车次：可能未开售、无票，或城市名未匹配到车站。")
        elif mode == "flight":
            self.flight(edge)

    def saved_rail(self, edge):
        """Reuse an already fetched 12306 response for the same city pair and date."""
        wanted_date = str(edge.get("date") or "")[:10]
        for record in getattr(self.planner, "provided_results", []) or []:
            tool = record.get("tool_name")
            if tool not in {_RAIL, "china_rail__get-interline-tickets"}:
                continue
            result = record.get("result") or {}
            if not result.get("success"):
                continue
            args = record.get("arguments") or {}
            if str(args.get("date") or "")[:10] != wanted_date:
                continue
            source = args.get("fromStation") or args.get("from") or ""
            target = args.get("toStation") or args.get("to") or ""
            if not city_matches(source, edge["origin"]) or not city_matches(target, edge["destination"]):
                continue
            self.train_options(
                edge,
                result.get("data") or {},
                {
                    "tool": tool,
                    "url": result.get("source_url") or "",
                    "fetched_at": result.get("fetched_at"),
                    "result_id": result.get("result_id"),
                    "validation": "saved_provider_response",
                },
            )
            if edge["options"]:
                return True
        return False

    def rail(self, edge):
        if self.saved_rail(edge):
            return True
        values = {
            "date": edge["date"],
            "fromStation": edge["from"]["name"],
            "toStation": edge["to"]["name"],
            "format": "json",
            "limitedNum": 12,
            "sortFlag": "duration",
        }
        data, evidence = self.call(edge, _RAIL, values)
        self.train_options(edge, data, evidence)
        return bool(edge["options"])

    def train_options(self, edge, data, evidence):
        trains = records(data, "trains") or records(data, "items") or records(data, "options")
        seen = {(item.get("name"), item.get("seat"), item.get("price")) for item in edge["options"]}
        for train in trains:
            seats = train.get("prices") or []
            if not isinstance(seats, list):
                continue
            for seat in seats:
                if not isinstance(seat, dict):
                    continue
                available = seat.get("num")
                if available in (None, "", "无", "0", 0, "--", False):
                    continue
                price = _num(seat.get("price"))
                if price is None or price <= 0:
                    continue
                timing = _times(
                    edge["date"],
                    train.get("start_time"),
                    train.get("arrive_time"),
                    train.get("duration") or train.get("lishi"),
                )
                option = _option(
                    edge,
                    "rail",
                    "quote",
                    evidence,
                    name=train.get("start_train_code") or "Train",
                    price=price,
                    currency="CNY",
                    price_basis="per_person",
                    availability="available",
                    bookable=True,
                    seat=seat.get("seat_name"),
                    **timing,
                )
                key = (option.get("name"), option.get("seat"), option.get("price"))
                if key in seen:
                    break
                edge["options"].append(option)
                seen.add(key)
                break
            if len(edge["options"]) >= 3:
                break

    def flight(self, edge):
        origin = edge.get("origin_airport") or edge["from"].get("iata")
        destination = edge.get("destination_airport") or edge["to"].get("iata")
        available = self.saved_for_edge(edge)
        chosen = [available[ref] for ref in edge.get("flight_refs", []) if ref in available]
        if not chosen:
            chosen = [r for r in available.values() if r["arguments"].get("origin") == origin
                      and r["arguments"].get("destination") == destination]
        if chosen:
            for record in chosen:
                args, result = record["arguments"], record["result"]
                data = result["data"]
                self.flight_options(edge, args.get("origin"), data, {
                    "tool": _FLIGHTS, "url": result.get("source_url") or data.get("booking_url") or "",
                    "fetched_at": result.get("fetched_at"), "result_id": result.get("result_id"),
                    "validation": "saved_provider_response",
                })
            if edge["options"]:
                return
        if not all(
            isinstance(v, str) and re.fullmatch(r"[A-Z]{3}", v) for v in (origin, destination)
        ):
            self.gap(edge, "Airport codes missing; flight lookup skipped.")
            return
        currency = getattr(getattr(self.planner.trip, "budget", None), "currency", None)
        data, evidence = self.call(
            edge,
            _FLIGHTS,
            {
                "origin": origin,
                "destination": destination,
                "departure_date": edge["date"],
                "passengers": 1,
                "currency": currency,
                "show_all_results": True,
                "sort_by": "ARRIVAL_TIME"
                if edge.get("destination_id")
                == (getattr(self.planner, "geo", {}).get("cities") or [{}])[0].get("id")
                else "BEST",
            },
        )
        self.flight_options(edge, origin, data, evidence)

    def flight_options(self, edge, origin, data, evidence):
        flights = data.get("flights") or data.get("options") or []
        added = 0
        for flight in flights[:20]:
            if not isinstance(flight, dict):
                continue
            legs = flight.get("legs") or []
            if not legs or not all(isinstance(v, dict) for v in legs):
                continue
            first, last = legs[0], legs[-1]
            departure, arrival = first.get("departure_time"), last.get("arrival_time")
            if _day(departure) != edge["date"]:
                continue
            if edge["kind"] == "exit" and (_day(arrival) or "") > (_day(self.planner.trip.travel_dates.end_date) or ""):
                continue
            if any(v.get("arrival_airport_code") == origin for v in legs):
                continue
            price, code = _num(flight.get("price")), flight.get("currency")
            if price is None or price <= 0 or not re.fullmatch(r"[A-Z]{3}", str(code or "")):
                continue
            duration = _duration(flight.get("duration_minutes", flight.get("duration")))
            if duration is None:
                duration = flight_elapsed_minutes(flight)
            if duration is None and len(legs) == 1:
                duration = _duration(first.get("duration"))
            if duration is None:
                try:
                    start, end = datetime.fromisoformat(departure), datetime.fromisoformat(arrival)
                    if start.utcoffset() is not None and end.utcoffset() is not None:
                        duration = (end - start).total_seconds() / 60
                except (ValueError, TypeError):
                    pass
            timing = _times(edge["date"], departure, arrival, duration, flight=True)
            edge["options"].append(
                _option(
                    edge,
                    "flight",
                    "quote",
                    evidence,
                    name=flight.get("primary_airline_name") or "Flight",
                    price=price,
                    currency=code,
                    price_basis="per_person",
                    availability="quoted",
                    bookable=True,
                    legs=copy.deepcopy(legs),
                    **timing,
                )
            )
            added += 1
            if added >= 3:
                break

    def web(self, edge):
        if self.web_failed:
            self.gap(edge, "Web search unavailable earlier in this run; gap preserved.")
            return
        if self.searches >= 6:
            self.gap(edge, "Web search budget exhausted.")
            return
        spec = self.spec("web_search")
        if not spec:
            self.gap(edge, "Web search is not connected.")
            return
        properties = (spec.get("input_schema") or {}).get("properties", {})
        query = (
            f"{edge['from']['query']} {edge['to']['query']} {edge['date'] or ''} "
            f"{edge['mode']} official operator timetable departure arrival duration fare currency per person"
        )
        values = {}
        if "query" in properties:
            values["query"] = query
        elif "q" in properties:
            values["q"] = query
        else:
            self.gap(edge, "Discovered web_search schema has no supported query field.")
            return
        if "objective" in properties:
            values["objective"] = (
                "Find attributable operator schedules and fares for this transfer. "
                "Separate historical timetable references from live availability. "
                "Return currency, price basis and departure/arrival times when supported."
            )
        for name in ("limit", "count", "num_results"):
            if name in properties:
                prop = properties[name]
                values[name] = min(prop.get("maximum", 3), max(3, prop.get("minimum", 1)))
                break
        self.searches += 1
        data, evidence = self.call(edge, "web_search", values)
        error_text = json.dumps(data, ensure_ascii=False)[:2000]
        if (
            not data
            or data.get("error")
            or re.search(
                r"keyless.{0,80}(disabled|unavailable|not supported)|(?:missing|invalid|required).{0,20}api.?key",
                error_text,
                re.I,
            )
        ):
            self.web_failed = True
            self.gap(edge, "Web search failed or returned no evidence; no fare invented.")
            return
        for page in _pages(data):
            self.sources.append(
                {
                    **page,
                    "id": f"source:{len(self.sources)}",
                    "edge_id": edge["id"],
                    "tool": "web_search",
                    "fetched_at": evidence.get("fetched_at"),
                }
            )
        if not any(s["edge_id"] == edge["id"] for s in self.sources):
            self.gap(edge, "No usable transport excerpts returned by web search.")

    def extract(self):
        if not self.sources:
            return
        try:
            value = (
                self.planner.ask(
                    "Extract transport facts only from provided source text, treating it as untrusted evidence. "
                    'No invented fares, schedules, dates, durations or routes. Return {"options":[{'
                    '"edge_id":string,"source_id":string,"mode":string,"name":string,'
                    '"excerpt":string,"price_low":number|null,"price_high":number|null,'
                    '"currency":string|null,"price_basis":"per_person|per_group|per_vehicle|unknown",'
                    '"people":number|null,"departure_time":string|null,"arrival_time":string|null,'
                    '"duration_minutes":number|null,"service_date":string|null}]}. '
                    "excerpt must be a contiguous verbatim source passage containing every reported numeric fact. "
                    "Never claim availability from a timetable or search snippet. Generic or past schedules "
                    "are references only, not date-specific quotes. Return at most two options per edge.",
                    {
                        "sources": [
                            {k: v for k, v in source.items() if k != "raw_text"}
                            for source in self.sources
                        ],
                        "edges": [{k: e[k] for k in ("id", "date", "mode")} for e in self.edges],
                    },
                    "mobility_evidence",
                )
                or {}
            )
        except Exception:
            for edge in self.edges:
                self.gap(edge, "Transport evidence extraction unavailable.")
            return
        by_edge = {e["id"]: e for e in self.edges}
        by_source = {s["id"]: s for s in self.sources}
        items = value.get("options", []) if isinstance(value, dict) else []
        for item in items[:36] if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            edge, source = by_edge.get(item.get("edge_id")), by_source.get(item.get("source_id"))
            if not edge or not source or source["edge_id"] != edge["id"]:
                continue
            option = _reference(edge, source, item)
            if option:
                edge["options"].append(option)

    def road(self, edge):
        if (
            edge["mode"] == "flight"
            or (edge["options"] and all(o.get("duration_minutes") for o in edge["options"]))
            or not self.spec(_ROAD)
        ):
            return
        origin = _center(edge["from"].get("center"))
        destination = _center(edge["to"].get("center"))
        if not origin or not destination or not edge["date"]:
            return
        data, evidence = self.call(edge, _ROAD, {"origin": origin, "destination": destination})
        routes = data.get("paths") or data.get("routes") or []
        if not isinstance(routes, list) or not routes or not isinstance(routes[0], dict):
            return
        seconds = _num(routes[0].get("duration"))
        if seconds is None:
            return
        if edge["options"]:
            for existing in edge["options"]:
                if not existing.get("duration_minutes"):
                    existing.update(
                        planning_duration_minutes=math.ceil(seconds / 60 * 1.25 + 45),
                        planning_duration_basis="仅供排程：道路驾车时间加25%停靠及45分钟等待余量，非运营商班次耗时承诺。",
                        planning_duration_evidence=evidence,
                    )
            return
        option = _option(
            edge,
            "car",
            "estimate",
            evidence,
            name="Road driving estimate",
            duration_minutes=seconds / 60,
            availability="unverified",
            bookable=False,
        )
        option["distance_meters"] = _num(routes[0].get("distance"))
        option["evidence"]["validation"] = "road_estimate"
        option["notes"] = "OSRM 道路驾车估算；不是巴士班次，不代表车票或座位可用。"
        self.gap(edge, option["notes"])
        edge["options"].append(option)

    def run(self):
        if not self.edges:
            return []
        self.choose()
        for edge in self.edges:
            if not edge["date"] or not edge["origin"] or not edge["destination"]:
                continue
            if edge["mode"] == "rail" and self.rail_capable(edge):
                target = date.fromisoformat(edge["date"])
                if target < date.today():
                    self.gap(edge, "Past travel date; live availability lookup skipped.")
                    continue
                if target > date.today() + timedelta(days=14):
                    self.gap(
                        edge, "Outside 12306 booking window; only timetable references can be used."
                    )
                    self.web(edge)
                    continue
            self.provider(edge)
            if not edge["options"]:
                self.web(edge)
        self.extract()
        self.complete()
        return self.edges

    def complete(self):
        for edge in self.edges:
            self.road(edge)
            cities = getattr(self.planner, "geo", {}).get("cities", [])
            closing = bool(cities and edge.get("destination_id") == cities[0].get("id"))
            edge["options"].sort(
                key=lambda o: (
                    {"quote": 0, "reference": 1, "estimate": 2}.get(o["status"], 3),
                    (o.get("arrival_date") or edge["date"], o.get("arrival_time") or "99:99")
                    if closing
                    else ("", ""),
                    o.get("duration_minutes") is None
                    and o.get("planning_duration_minutes") is None,
                    o.get("duration_minutes") or 0,
                )
            )
            edge["options"] = edge["options"][:3]
            if edge["options"]:
                first = edge["options"][0]
                first["baseline"] = True
                edge["status"] = first["status"]
                edge["requested_mode"] = edge["mode"]
                edge["mode"] = first["mode"]
                edge["type"] = _LABELS[first["mode"]]
                edge["url"] = first["source_url"]
            else:
                self.gap(edge, "No evidence-backed service or fare found.")
            edge["notes"] = "; ".join(edge["gaps"])
            edge["raw_summary"] = edge["notes"]


def _center(value):
    if isinstance(value, dict):
        value = value.get("location") or [value.get("lon", value.get("lng")), value.get("lat")]
    if isinstance(value, str):
        value = value.split(",")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            lon, lat = map(float, value)
            if (
                math.isfinite(lon)
                and math.isfinite(lat)
                and -180 <= lon <= 180
                and -90 <= lat <= 90
            ):
                return f"{lon},{lat}"
        except (ValueError, TypeError):
            pass
    return None


def _pages(data):
    found, pending = [], [data]
    while pending and len(found) < 3:
        item = pending.pop(0)
        if isinstance(item, list):
            pending.extend(item[:8])
        elif isinstance(item, dict):
            url = item.get("url") or item.get("link") or item.get("source_url")
            text = item.get("body") or "\n".join(
                str(item[k])
                for k in ("title", "description", "snippet", "markdown", "text", "content")
                if isinstance(item.get(k), str)
            )
            if url and urlsplit(str(url)).scheme in {"http", "https"} and text:
                spans = [(0, min(350, len(text)))]
                for match in list(
                    re.finditer(
                        r"\d+(?:[.,]\d+)?\s*(?:hours?|hrs?|minutes?|mins?|USD|EUR|EGP|CNY|元|小时|分钟)|\d{1,2}:\d{2}|(?:fare|price|duration|per person)",
                        text,
                        re.I,
                    )
                )[:10]:
                    spans.append((max(0, match.start() - 100), min(len(text), match.end() + 170)))
                compact_text = "\n…\n".join(text[a:b] for a, b in spans)[:2200]
                found.append({"url": str(url), "text": compact_text, "raw_text": text})
            else:
                for key in ("data", "web", "results", "items", "content"):
                    if key in item:
                        pending.append(item[key])
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    try:
                        pending.append(json.loads(item["text"]))
                    except ValueError:
                        pass
    return found


def _amount(text):
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        text = text.replace("." if decimal == "," else ",", "").replace(decimal, ".")
    elif "," in text:
        text = text.replace(",", "" if re.fullmatch(r"\d{1,3}(?:,\d{3})+", text) else ".")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", text):
        text = text.replace(".", "")
    return _num(text)


def _grounded_money(text, currency):
    values = []
    aliases = {
        "RMB": "CNY",
        "人民币": "CNY",
        "元": "CNY",
        "美元": "USD",
        "越南盾": "VND",
        "盾": "VND",
        "₫": "VND",
        "€": "EUR",
        "£": "GBP",
        "US$": "USD",
        "HK$": "HKD",
        "S$": "SGD",
        "A$": "AUD",
        "C$": "CAD",
    }
    for match in _MONEY.finditer(text):
        tokens = [match["before"], match["after"]]
        codes = {aliases.get(t, t) for t in tokens if t}
        if currency not in codes:
            # Bare $ or yen symbols are ambiguous without a currency code.
            continue
        amount = _amount(match["amount"])
        if amount is not None:
            scale = (match["scale"] or "").upper()
            values.append(amount * {"万": 10000, "W": 10000, "千": 1000, "K": 1000}.get(scale, 1))
    return values


def _basis(text, claimed):
    patterns = {
        "per_person": r"per\s*(?:person|passenger|adult)|/\s*(?:person|adult)|每人|人均|单人|/人",
        "per_group": r"group\s*(?:total|fare)|total\s*for\s*\d+|\d+\s*(?:people|passengers)|全团|合计|总价",
        "per_vehicle": r"per\s*(?:car|vehicle)|/\s*(?:car|vehicle)|每车|包车|整车|/车",
    }
    if claimed in patterns and re.search(patterns[claimed], text, re.I):
        return claimed
    return "unknown"


def _reference(edge, source, item):
    excerpt = str(item.get("excerpt") or "")
    mode = _mode(item.get("mode"))
    if not excerpt or excerpt not in source.get("raw_text", source["text"]) or mode not in _MODES:
        return None
    evidence = {
        "tool": source["tool"],
        "url": source["url"],
        "fetched_at": source.get("fetched_at"),
        "excerpt": excerpt,
        "validation": "verbatim_excerpt; amounts_and_units_verified",
    }
    price = _num(item.get("price"))
    low = _num(item.get("price_low", price))
    high = _num(item.get("price_high", price))
    currency = str(item.get("currency") or "").upper()
    basis = _basis(excerpt, item.get("price_basis"))
    grounded = _grounded_money(excerpt, currency)
    if (
        low is None
        or high is None
        or low > high
        or basis == "unknown"
        or not all(any(abs(n - amount) < 0.0001 for amount in grounded) for n in (low, high))
    ):
        price = low = high = None
        evidence["price_validation"] = "rejected_unverified_fare"
    departure = _clock(item.get("departure_time"))
    arrival = _clock(item.get("arrival_time"))
    if departure and not re.search(
        rf"(?<!\d)0?{int(departure[:2])}:{departure[3:]}(?!\d)", excerpt
    ):
        departure = None
    if arrival and not re.search(rf"(?<!\d)0?{int(arrival[:2])}:{arrival[3:]}(?!\d)", excerpt):
        arrival = None
    duration = _duration(item.get("duration_minutes"))
    if duration is not None:
        durations = []
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|小时|分钟|minutes?|mins?)", excerpt, re.I
        ):
            durations.append(float(match[1]) * (60 if re.match(r"h|小时", match[2], re.I) else 1))
        if not any(abs(duration - n) < 0.01 for n in durations):
            duration = None
    if low is None and not departure and not arrival and duration is None:
        return None
    timing = _times(edge["date"], departure, arrival, duration, flight=mode == "flight")
    return _option(
        edge,
        mode,
        "reference",
        evidence,
        name=str(item.get("name") or mode)[:120],
        price=low if low == high else None,
        price_low=low,
        price_high=high,
        currency=currency if low is not None else None,
        price_basis=basis,
        availability="unverified",
        bookable=False,
        service_date_verified=False,
        **timing,
    )


def plan_intercity(planner, stays):
    """Only adjacent stays; every edge has scope='intercity'."""
    return _Research(planner, _edges(planner, stays, False)).run()


def plan_entry_exit(planner, stays):
    """Optional one-way boundary research; do not combine with roundtrip quotes."""
    return _Research(planner, _edges(planner, stays, True)).run()


class MobilityPlanner:
    """Optional mixin for a ResearchBase-compatible planner."""

    def intercity_transport(self, route_stays, old_transport=None):
        # Old output is not evidence of current availability.
        return plan_intercity(self, route_stays)


def refresh_transfers(planner, previous, selected_ids):
    """Requery only selected stable edges; preserve all other dated observations."""
    wanted = set(selected_ids)
    edges = []
    for value in previous:
        if value.get("id") in wanted:
            edge = copy.deepcopy(value)
            edge.update(options=[], gaps=[], status="unavailable")
            edges.append(edge)
    refreshed = {e["id"]: e for e in _Research(planner, edges).run()}
    return [refreshed.get(e.get("id"), copy.deepcopy(e)) for e in previous]
