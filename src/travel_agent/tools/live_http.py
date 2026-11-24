"""Small, read-only live tools for international maps and currency conversion."""

from __future__ import annotations

import os
import re
import time
from datetime import UTC, datetime, timedelta

import httpx

from travel_agent.models import ToolResult, ToolSpec
from travel_agent.tools.base import TravelTool

# Photon reports Hong Kong/Macau features under the mainland China country code.
COUNTRY_ALIASES = {"HK": {"CN"}, "MO": {"CN"}}
# (min_lon, min_lat, max_lon, max_lat) used to keep CN-indexed results in-region.
REGION_BOUNDS = {
    "HK": (113.82, 22.14, 114.44, 22.57),
    "MO": (113.52, 22.10, 113.60, 22.22),
}


def country_matches(actual: str | None, wanted: str | None) -> bool:
    actual_code = (actual or "").upper()
    wanted_code = (wanted or "").upper()
    return actual_code == wanted_code or actual_code in COUNTRY_ALIASES.get(wanted_code, set())


def _inside(coordinates, bounds) -> bool:
    lon, lat = float(coordinates[0]), float(coordinates[1])
    min_lon, min_lat, max_lon, max_lat = bounds
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _center_pair(center: str) -> tuple[float, float]:
    """Accept ``lon,lat`` or ``lat,lon`` and return bounded ``(lon, lat)``.

    Models sometimes pass coordinates in latitude-first order; the bounds check
    keeps the bbox valid so Photon does not reject it with HTTP 400.
    """
    try:
        first, second = (float(part) for part in str(center).split(",", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("center must be 'longitude,latitude' in WGS84") from exc
    lon, lat = (first, second) if abs(second) <= 90 else (second, first)
    if abs(lon) > 180 or abs(lat) > 90:
        raise ValueError("center coordinates out of WGS84 range")
    return lon, max(-89.9, min(89.9, lat))


class LiveHTTPTool(TravelTool):
    source_type = "api"
    max_retries = 1
    _last_geocode = 0.0

    def __init__(self, name, description, properties, required, operation):
        self.operation = operation
        self.spec = ToolSpec(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        )

    def run(self, arguments):
        try:
            with httpx.Client(
                timeout=25,
                follow_redirects=True,
                headers={
                    "User-Agent": "TravelPlanningAgent/0.2 (local academic travel planner)",
                },
            ) as client:
                data, url = getattr(self, self.operation)(client, **arguments)
            now = datetime.now(UTC)
            return ToolResult(
                success=True,
                tool_name=self.spec.name,
                data=data,
                summary=str(data)[:1200],
                source_type="api",
                source_name=self.operation,
                source_url=url,
                fetched_at=now,
                valid_until=now + timedelta(hours=12),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                tool_name=self.spec.name,
                source_type="api",
                source_name=self.operation,
                error_type="http_error",
                error_message=str(exc),
            )

    def geocode(self, client, query, country_code, center=None):
        # Low rate, bounded individual searches on Photon; no area-wide scraping.
        pause = 1.1 - (time.monotonic() - LiveHTTPTool._last_geocode)
        if pause > 0:
            time.sleep(pause)
        wanted = (country_code or "").upper()
        region = REGION_BOUNDS.get(wanted)
        params = {
            "q": query,
            "countrycode": "CN" if region else wanted,
            "limit": 10,
            "lang": "en",
        }
        if center:
            lon, lat = _center_pair(center)
            params["bbox"] = (
                f"{lon - 1.2:.4f},{lat - 1.2:.4f},{lon + 1.2:.4f},{lat + 1.2:.4f}"
            )
            # A centre bias can rank a nearby bus stop ahead of the named attraction.
        elif region:
            # Photon indexes Hong Kong/Macau under CN, so bound the search instead.
            params["bbox"] = ",".join(f"{value:.4f}" for value in region)
        url = os.getenv("TRAVEL_AGENT_PHOTON_URL", "https://photon.komoot.io/api/")
        response = client.get(url, params=params)
        if response.status_code == 400 and "bbox" in params:
            # Photon rejects out-of-range bounds with 400; drop the bias and retry.
            params.pop("bbox", None)
            response = client.get(url, params=params)
        LiveHTTPTool._last_geocode = time.monotonic()
        response.raise_for_status()
        found = []
        for feature in response.json().get("features", []):
            props = feature.get("properties", {})
            if not country_matches(props.get("countrycode"), wanted):
                continue
            coordinates = feature.get("geometry", {}).get("coordinates", [])
            if len(coordinates) != 2:
                continue
            if region and not _inside(coordinates, region):
                continue
            osm_type = {"N": "node", "W": "way", "R": "relation"}.get(props.get("osm_type"))
            found.append(
                {
                    **props,
                    "id": f"osm:{osm_type}:{props.get('osm_id')}",
                    "location": ",".join(map(str, coordinates)),
                    "country_code": country_code.upper(),
                    "category": props.get("osm_value"),
                    "coordinate_system": "WGS84",
                    "source_url": f"https://www.openstreetmap.org/{osm_type}/{props.get('osm_id')}",
                }
            )
        return {
            "places": found,
            "attribution": "© OpenStreetMap contributors; Photon / komoot",
        }, str(response.url)

    def route(self, client, origin, destination):
        for value in (origin, destination):
            if not re.fullmatch(r"-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?", value):
                raise ValueError("Expected longitude,latitude in WGS84")
        base = os.getenv("TRAVEL_AGENT_OSRM_URL", "https://router.project-osrm.org")
        response = client.get(
            f"{base}/route/v1/driving/{origin};{destination}", params={"overview": "false"}
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code") != "Ok":
            raise ValueError(data.get("message", "Route not found"))
        # Excessive snapping can silently route from a distant road (islands, parks).
        if any(p.get("distance", 0) > 1500 for p in data.get("waypoints", [])):
            raise ValueError("Route endpoint is more than 1.5 km from the requested point")
        return {
            "paths": data["routes"],
            "attribution": "© OpenStreetMap contributors; OSRM",
            "traffic": "道路模型估算，不含实时拥堵",
        }, str(response.url)

    def exchange_rates(self, client, base):
        if not re.fullmatch("[A-Z]{3}", base):
            raise ValueError("Expected ISO currency")
        response = client.get(f"https://open.er-api.com/v6/latest/{base}")
        response.raise_for_status()
        data = response.json()
        if data.get("result") != "success":
            raise ValueError("Exchange-rate service unavailable")
        return {
            "base": base,
            "rates": data["rates"],
            "updated_at": data["time_last_update_utc"],
            "provider": data["provider"],
        }, str(response.url)


def build_live_http_tools():
    string = {"type": "string"}
    return [
        LiveHTTPTool(
            "open_maps__search_places",
            "查询境外真实地点与坐标；必须给国家代码，建议提供目的地中心坐标限制同名地点。使用 Photon / OpenStreetMap。",
            {"query": string, "country_code": string, "center": string},
            ["query", "country_code"],
            "geocode",
        ),
        LiveHTTPTool(
            "open_maps__driving_route",
            "查询境外两个 WGS84 坐标之间的道路距离及驾车模型耗时；不是实时交通或打车报价。",
            {"origin": string, "destination": string},
            ["origin", "destination"],
            "route",
        ),
        LiveHTTPTool(
            "research__exchange_rates",
            "查询汇率与更新时间，base 为三字母币种代码，用于有来源的旅行预算换算。",
            {"base": string},
            ["base"],
            "exchange_rates",
        ),
    ]
