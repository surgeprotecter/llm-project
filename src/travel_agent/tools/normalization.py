from __future__ import annotations

import re
from typing import Any


def flight_elapsed_minutes(flight: dict[str, Any]) -> int | float | None:
    legs, stops = flight.get("legs") or [], flight.get("layovers") or []
    if not legs or len(stops) != len(legs) - 1:
        return None
    parts = [part.get("duration") for part in [*legs, *stops] if isinstance(part, dict)]
    if len(parts) != len(legs) + len(stops) or not all(type(value) in (int, float) and value >= 0 for value in parts):
        return None
    return sum(parts)


def rail_text_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse provider text into complete records, never split a fare row in half."""
    text = data.get("items")
    if not isinstance(text, str):
        return []
    records = []
    pattern = re.compile(
        r"^([A-Z]\d+)\s+(.+?)\(telecode:([A-Z]+)\)\s*->\s*"
        r"(.+?)\(telecode:([A-Z]+)\)\s+(\d\d:\d\d)\s*->\s*(\d\d:\d\d)\s+历时[：:](\d\d:\d\d)",
        re.M,
    )
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        code, origin, origin_code, destination, destination_code, departure, arrival, duration = match.groups()
        block = text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        seats = {
            seat: {"availability": availability.strip(), "price_cny_per_person": float(price)}
            for seat, availability, price in re.findall(r"^-\s*([^:\n]+):\s*(.*?)\s+(\d+(?:\.\d+)?)元\s*$", block, re.M)
        }
        records.append({
            "train": code, "origin": origin, "origin_code": origin_code,
            "destination": destination, "destination_code": destination_code,
            "departure_time": departure, "arrival_time": arrival, "duration": duration,
            "seats": seats,
        })
    return records


def _amap_transit_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    transits = data.get("transits")
    if not isinstance(transits, list):
        return []
    candidates = []
    for index, transit in enumerate(transits[:10], start=1):
        if not isinstance(transit, dict):
            continue
        line_names: list[str] = []
        for segment in transit.get("segments", []):
            if not isinstance(segment, dict):
                continue
            bus = segment.get("bus")
            if isinstance(bus, dict):
                for line in bus.get("buslines", []):
                    if isinstance(line, dict) and isinstance(line.get("name"), str):
                        line_names.append(line["name"])
            railway = segment.get("railway")
            if isinstance(railway, dict) and isinstance(railway.get("name"), str) and railway["name"].strip():
                line_names.append(railway["name"])
        details = []
        try:
            seconds = int(float(transit.get("duration", 0)))
        except (TypeError, ValueError):
            seconds = 0
        if seconds > 0:
            minutes = round(seconds / 60)
            details.append(f"约{minutes // 60}小时{minutes % 60}分钟" if minutes >= 60 else f"约{minutes}分钟")
        for key, label in (("distance", "全程"), ("walking_distance", "步行")):
            try:
                meters = float(transit.get(key, 0))
            except (TypeError, ValueError):
                meters = 0
            if meters > 0:
                details.append(f"{label}{meters / 1000:.1f}公里")
        if line_names:
            details.append("途经" + "、".join(dict.fromkeys(line_names[:8])))
        origin = str(data.get("origin") or "未知起点坐标")
        destination = str(data.get("destination") or "未知终点坐标")
        candidates.append(
            {
                "route": f"高德公共交通候选 {index}（{origin} → {destination}）",
                "durationText": "；".join(details) or "路线服务未返回用时详情",
                "origin_coordinate": origin,
                "destination_coordinate": destination,
                "distance_m": transit.get("distance", data.get("distance")),
                "walking_distance_m": transit.get("walking_distance"),
                "transit_lines": list(dict.fromkeys(line_names)),
            }
        )
    return candidates


def _find_records(value: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records = [item for item in value if isinstance(item, dict)]
        if records:
            return records
        for item in value:
            found = _find_records(item, keys)
            if found:
                return found
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, list):
                records = [item for item in candidate if isinstance(item, dict)]
                if records:
                    return records
        for candidate in value.values():
            if isinstance(candidate, (dict, list)):
                found = _find_records(candidate, keys)
                if found:
                    return found
    return []


def normalize_travel_data(tool_name: str, source_name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Add a small common shape while preserving provider fields and provenance."""
    normalized = dict(data)
    name = f"{source_name} {tool_name}".lower()

    if "xiaohongshu" in name or "xhs" in name:
        notes = _find_records(data, ("feeds", "notes", "items", "results", "data"))
        if notes:
            normalized["research_notes"] = notes[:12]

    if "amap" in name or "maps" in name:
        if any(
            token in tool_name.lower()
            for token in ("search_places", "resolve_names", "maps_text_search", "maps_around_search", "maps_search_detail")
        ):
            places = _find_records(data, ("places", "pois", "results", "items", "data"))
            if places:
                for place in places:
                    place.setdefault(
                        "source_url",
                        place.get("amap_uri") or place.get("maps_url") or place.get("uri") or place.get("url"),
                    )
                normalized["attractions"] = places[:20]
        if any(token in tool_name.lower() for token in ("compute_routes", "maps_direction_", "maps_distance")):
            routes = (
                _amap_transit_candidates(data)
                if "maps_direction_transit_integrated" in tool_name.lower()
                else _find_records(data, ("routes", "paths", "transits", "legs", "results", "data"))
            )
            if routes:
                normalized["route"] = routes[:20]

    if "booking" in name:
        hotels = _find_records(data, ("properties", "hotels", "results", "items", "data"))
        if hotels:
            normalized["hotels"] = hotels[:20]
            for hotel in normalized["hotels"]:
                if hotel.get("priceIsTotal"):
                    hotel["total_price"] = hotel.get("totalPrice", hotel.get("pricePerNight", hotel.get("price")))
                    hotel.pop("nightly_price", None)
                elif "nightly_price" not in hotel:
                    hotel["nightly_price"] = hotel.get("pricePerNight", hotel.get("price"))
                hotel.setdefault("source_url", hotel.get("url") or hotel.get("propertyUrl"))

    if "fli" in name or "flight" in name:
        flights = _find_records(data, ("flights", "options", "results", "items", "data"))
        if flights:
            options = []
            for flight in flights[:20]:
                option = dict(flight)
                legs = flight.get("legs", [])
                if not isinstance(legs, list):
                    legs = []
                if not legs and not any(
                    flight.get(key)
                    for key in ("route", "flight_number", "primary_airline", "primary_airline_name", "airline")
                ):
                    continue
                if isinstance(legs, list) and legs:
                    first = legs[0] if isinstance(legs[0], dict) else {}
                    last = legs[-1] if isinstance(legs[-1], dict) else first
                    origin = first.get("departure_airport_name") or first.get("departure_airport")
                    # A round-trip result contains both directions in `legs`.
                    # Its route label is the outbound destination, not the final home airport.
                    outward = next((leg for i,leg in enumerate(legs[:-1]) if str(legs[i+1].get("departure_time", ""))[:10] > str(leg.get("arrival_time", ""))[:10]), last)
                    destination = outward.get("arrival_airport_name") or outward.get("arrival_airport")
                    airline = flight.get("primary_airline_name") or flight.get("primary_airline") or first.get("airline")
                    flight_number = first.get("flight_number")
                    if origin and destination:
                        route = f"{origin} → {destination}"
                        if airline:
                            route += f" · {airline}"
                        if flight_number:
                            route += f" {flight_number}"
                        option.setdefault("route", route)
                    if first.get("departure_time"):
                        option.setdefault("departure_time", first["departure_time"])
                    if outward.get("arrival_time"):
                        option.setdefault("arrival_time", outward["arrival_time"])
                    duration = first.get("duration")
                    if isinstance(duration, (int, float)):
                        times = ""
                        if first.get("departure_time") and outward.get("arrival_time"):
                            times = f"{str(first['departure_time'])[-8:-3]}–{str(outward['arrival_time'])[-8:-3]} · "
                        option.setdefault("durationText", f"{times}{int(duration)} 分钟")
                    option.setdefault("flightNumber", flight_number)
                option.setdefault("airline", airline if legs else flight.get("primary_airline_name") or flight.get("primary_airline"))
                option.setdefault("name", flight.get("primary_airline_name") or flight.get("primary_airline") or "航班")
                option.setdefault("estimated_price", flight.get("price"))
                option.setdefault("currency", flight.get("currency"))
                options.append(option)
            normalized["options"] = options

    if "12306" in name or "rail" in name or "train" in name:
        trains = _find_records(data, ("trains", "tickets", "results", "items", "data"))
        if trains:
            normalized["options"] = trains[:20]

    return normalized
