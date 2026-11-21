from __future__ import annotations

from travel_agent.models import ToolResult


def test_unknown_tool_is_bounded(database) -> None:
    from travel_agent.tools import ToolRegistry

    registry = ToolRegistry(database)
    result = registry.run("not_registered", {}, "u1:t1")
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.error_type == "unknown_tool"
