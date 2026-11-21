"""Small, read-only live tools for international maps and currency conversion."""

from __future__ import annotations

import os
import re
import time
from datetime import UTC, datetime, timedelta

import httpx

from travel_agent.models import ToolResult, ToolSpec
from travel_agent.tools.base import TravelTool


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
        params = {"q": query, "countrycode": country_code.upper(), "limit": 10, "lang": "en"}
        if center:
            lon, lat = map(float, center.split(","))
            params["bbox"] = f"{lon - 1.2},{lat - 1.2},{lon + 1.2},{lat + 1.2}"
            # A centre bias can rank a nearby bus stop ahead of the named attraction.
        url = os.getenv("TRAVEL_AGENT_PHOTON_URL", "https://photon.komoot.io/api/")
        response = client.get(url, params=params)
        LiveHTTPTool._last_geocode = time.monotonic()
        response.raise_for_status()
        found = []
        for feature in response.json().get("features", []):
            props = feature.get("properties", {})
            if props.get("countrycode", "").upper() != country_code.upper():
                continue
            coordinates = feature.get("geometry", {}).get("coordinates", [])
            if len(coordinates) != 2:
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
