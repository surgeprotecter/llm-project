from __future__ import annotations

from datetime import timedelta
from typing import Any

from travel_agent.mcp.manager import MCPServerManager
from travel_agent.models import ToolResult, ToolSpec
from travel_agent.tools.base import TravelTool


class MCPToolAdapter(TravelTool):
    """Expose one discovered MCP tool through the application's Tool protocol."""

    def __init__(self, manager: MCPServerManager, metadata: dict[str, Any]):
        self.manager = manager
        self.server_name = metadata["server_name"]
        self.remote_name = metadata["remote_name"]
        self.max_retries = metadata["config"].max_retries
        self.spec = ToolSpec(
            name=metadata["exposed_name"],
            description=metadata["description"],
            input_schema=metadata["input_schema"],
            output_schema=metadata["output_schema"],
        )
        if self.spec.name == "google_flights__search_flights":
            self.spec.description += (
                " Price is the full itinerary total for the search passenger party, not per person; "
                "round-trip price covers outbound and return together. Divide by the number of "
                "travelers only to show an average per person."
            )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        invocation = self.manager.call_tool(self.server_name, self.remote_name, arguments)
        fetched_at = invocation.fetched_at
        data = invocation.data
        if self.spec.name == "google_flights__search_flights" and invocation.success:
            party_size = sum(int(arguments.get(k) or 0) for k in
                             ("passengers", "children", "infants_in_seat", "infants_on_lap")) or 1
            data = {**data, "quote_context": {
                "passengers": party_size, "price_unit": "total_for_search_party",
                "trip_type": "round_trip" if arguments.get("return_date") else "one_way",
                "note": "返回price为查询人数的全行程合计；往返价格不能再将两程相加。",
            }}
        return ToolResult(
            success=invocation.success,
            tool_name=self.spec.name,
            data=data,
            summary=invocation.summary,
            warnings=invocation.warnings or [],
            source_type="mcp",
            source_name=self.server_name,
            source_url=_source_url(invocation.data),
            fetched_at=fetched_at,
            valid_until=fetched_at + timedelta(minutes=30),
            error_type=invocation.error_type,
            error_message=invocation.error_message,
        )


def _source_url(data: dict[str, Any]) -> str | None:
    keys = ("source_url", "url", "link", "web_url", "bookingUrl", "maps_url", "amap_uri")
    pending: list[Any] = [data]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key in keys:
                value = current.get(key)
                if isinstance(value, str) and value.startswith(("https://", "http://")):
                    return value
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return None
