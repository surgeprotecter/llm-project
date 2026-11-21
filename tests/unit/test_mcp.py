from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from travel_agent.mcp import (
    MCPInvocation,
    MCPServerConfig,
    MCPServerManager,
    MCPToolAdapter,
    load_mcp_configs,
)
from travel_agent.mcp.manager import _headers, _normalise_result, _resolved_url
from travel_agent.models import ToolSpec
from travel_agent.tools.base import TravelTool
from travel_agent.tools.registry import ToolRegistry


@pytest.mark.parametrize("data,expected", [
    ({"success": False, "flights": [], "error": "No shopping payload", "error_type": "search_error"}, False),
    ({"success": True, "flights": [], "count": 0}, True),
])
def test_mcp_manager_respects_embedded_service_failure(monkeypatch, data, expected) -> None:
    class Session:
        async def call_tool(self, name, arguments):
            return CallToolResult(content=[], structuredContent=data, isError=False)

    @asynccontextmanager
    async def session(config):
        yield Session()

    manager = MCPServerManager([MCPServerConfig(name="google_flights", command="unused")])
    monkeypatch.setattr(manager, "_session", session)
    result = manager.call_tool("google_flights", "search_flights", {})
    assert result.success is expected
    if not expected:
        assert result.error_type == "search_error"
        assert result.error_message == "No shopping payload"


def test_mcp_config_loads_allowlist_and_alias(tmp_path) -> None:
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        '{"servers": [{"name": "brave", "command": "npx", "allowed_tools": ["brave_web_search"], "aliases": {"brave_web_search": "web_search"}}]}',
        encoding="utf-8",
    )

    configs = load_mcp_configs(config_path)

    assert configs[0].name == "brave"
    assert configs[0].allowed_tools == ["brave_web_search"]
    assert configs[0].aliases["brave_web_search"] == "web_search"


def test_mcp_result_normalisation_handles_json_text() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text='{"results": [{"title": "成都"}]}')],
        isError=False,
    )

    data, summary = _normalise_result(result)

    assert data["results"][0]["title"] == "成都"
    assert "成都" in summary


def test_mcp_adapter_maps_external_result_to_project_protocol() -> None:
    class FakeManager:
        def call_tool(self, server_name, remote_name, arguments):
            assert (server_name, remote_name) == ("brave", "brave_web_search")
            assert arguments["query"] == "成都旅行"
            return MCPInvocation(
                success=True,
                data={"results": [{"title": "成都旅行攻略"}]},
                summary="搜索完成",
                fetched_at=datetime.now(UTC),
            )

    config = MCPServerConfig(name="brave", command="npx", aliases={"brave_web_search": "web_search"})
    adapter = MCPToolAdapter(
        FakeManager(),
        {
            "server_name": "brave",
            "remote_name": "brave_web_search",
            "exposed_name": "web_search",
            "description": "搜索网页",
            "input_schema": {"type": "object"},
            "output_schema": {},
            "config": config,
        },
    )

    result = adapter.run({"query": "成都旅行"})

    assert adapter.spec.name == "web_search"
    assert result.success
    assert result.source_type == "mcp"
    assert result.data["results"][0]["title"] == "成都旅行攻略"


def test_mcp_manager_discovers_and_calls_stdio_server() -> None:
    server_path = Path(__file__).parents[1] / "fixtures" / "mcp_demo_server.py"
    config = MCPServerConfig(
        name="demo",
        command=sys.executable,
        args=[str(server_path)],
        allowed_tools=["brave_web_search"],
        aliases={"brave_web_search": "web_search"},
    )
    manager = MCPServerManager([config])

    discovered = manager.discover_tools()
    result = manager.call_tool("demo", "brave_web_search", {"query": "成都", "count": 1})

    assert [item["exposed_name"] for item in discovered] == ["web_search"]
    assert result.success
    assert result.data["results"][0]["title"] == "Result for 成都"


def test_missing_http_credential_skips_discovery_without_network(monkeypatch) -> None:
    monkeypatch.delenv("TRAVEL_AGENT_TEST_MCP_TOKEN", raising=False)
    config = MCPServerConfig(
        name="secured",
        transport="streamable_http",
        url="https://example.invalid/mcp",
        headers={"Authorization": "Bearer ${TRAVEL_AGENT_TEST_MCP_TOKEN}"},
    )
    manager = MCPServerManager([config])

    assert manager.discover_tools() == []
    assert "TRAVEL_AGENT_TEST_MCP_TOKEN" in manager.discovery_errors["secured"]


def test_optional_http_credential_is_omitted_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("TRAVEL_AGENT_OPTIONAL_TOKEN", raising=False)
    config = MCPServerConfig(
        name="optional-auth",
        transport="streamable_http",
        url="https://example.invalid/mcp",
        headers={"Authorization": "Bearer ${TRAVEL_AGENT_OPTIONAL_TOKEN}"},
        optional_headers=["Authorization"],
    )

    assert _headers(config) == {}
    monkeypatch.setenv("TRAVEL_AGENT_OPTIONAL_TOKEN", "test-only")
    assert _headers(config) == {"Authorization": "Bearer test-only"}


def test_http_url_expands_api_key_and_redacts_it_from_errors(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_TEST_AMAP_KEY", "local-test-key")
    config = MCPServerConfig(
        name="amap",
        transport="streamable_http",
        url="https://mcp.amap.com/mcp?key=${TRAVEL_AGENT_TEST_AMAP_KEY}",
    )
    assert _resolved_url(config) == "https://mcp.amap.com/mcp?key=local-test-key"

    manager = MCPServerManager([config])

    async def fail_with_url(server_config):
        raise RuntimeError(_resolved_url(server_config))

    manager._discover_server = fail_with_url
    assert manager.discover_tools() == []
    assert "<redacted>" in manager.discovery_errors["amap"]
    assert "local-test-key" not in manager.discovery_errors["amap"]


def test_registry_rejects_missing_required_arguments(database) -> None:
    class RequiredInputTool(TravelTool):
        spec = ToolSpec(
            name="required_input",
            description="test",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )

        def run(self, arguments):
            raise AssertionError("invalid arguments must never reach the tool")

    registry = ToolRegistry(database)
    registry.register(RequiredInputTool())

    result = registry.run("required_input", {}, "u:t")

    assert not result.success
    assert result.error_type == "invalid_arguments"
