"""Read explicit room-price context that the upstream MCP search drops."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from travel_agent.models import ToolResult, ToolSpec
from travel_agent.tools.base import TravelTool


class HotelQuoteTool(TravelTool):
    source_type = "cli"
    max_retries = 0
    spec = ToolSpec(
        name="booking_evidence__read_quote",
        description="只读酒店详情的价格上下文、房型行、币种与图片；用于核对每晚/入住总价，不执行预订。",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "name": {"type": "string"},
                "destination": {"type": "string"},
                "checkin": {"type": "string"},
                "checkout": {"type": "string"},
                "adults": {"type": "integer", "minimum": 1},
                "rooms": {"type": "integer", "minimum": 1},
                "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
            },
            "required": ["url", "checkin", "checkout", "adults", "rooms", "currency"],
            "additionalProperties": False,
        },
    )

    def run(self, arguments):
        script = Path(__file__).resolve().parents[3] / "scripts" / "read_hotel_quote.cjs"
        process = subprocess.run(
            ["node", str(script)],
            input=json.dumps(arguments),
            capture_output=True,
            text=True,
            timeout=90,
        )
        data = json.loads(process.stdout)
        now = datetime.now(UTC)
        return ToolResult(
            success=process.returncode == 0 and not data.get("error"),
            tool_name=self.spec.name,
            data=data,
            summary=data.get("title", data.get("error", "")),
            source_type="cli",
            source_name="booking_visible_quote",
            source_url=arguments["url"],
            fetched_at=now,
            valid_until=now + timedelta(minutes=30),
            error_type="quote_read_error" if data.get("error") else None,
        )
