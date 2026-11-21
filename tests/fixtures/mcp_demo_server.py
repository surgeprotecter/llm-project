from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer

server = MCPServer("travel-agent-test")


@server.tool(name="brave_web_search", description="Test web search tool")
def brave_web_search(query: str, count: int = 5) -> dict[str, Any]:
    return {"results": [{"title": f"Result for {query}", "url": "https://example.com", "count": count}]}


if __name__ == "__main__":
    asyncio.run(server.run_stdio_async())
