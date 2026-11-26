from __future__ import annotations

from datetime import UTC, datetime, timedelta
from io import StringIO

from rich.console import Console

from travel_agent.cli import _print_trace, _trace_redact
from travel_agent.llm import ModelGateway
from travel_agent.models import ToolExecution, ToolResult


def test_full_trace_prints_model_and_tool_io(database, settings, monkeypatch) -> None:
    gateway = ModelGateway(settings)
    gateway.call_count = 1
    gateway.success_count = 1
    gateway.last_status = "success"
    gateway.attempt_trace = [
        {
            "stage": "plan",
            "attempt": 1,
            "started_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
            "duration_ms": 123,
            "prompt_chars": 18,
            "status": "success",
            "input_messages": [
                {"role": "system", "content": "选择工具"},
                {"role": "human", "content": "去岘港"},
            ],
            "output": '{"tool_name":"web_search"}',
        }
    ]
    database.save_tool_execution(
        ToolExecution(
            thread_id="u1:t1",
            tool_name="web_search",
            arguments={"query": "岘港景点"},
            result=ToolResult(
                success=True,
                tool_name="web_search",
                data={"items": ["美溪海滩"]},
                source_type="mcp",
                source_name="search_server",
            ),
        )
    )
    output = StringIO()
    monkeypatch.setattr("travel_agent.cli.console", Console(file=output, force_terminal=False, width=200))

    _print_trace(database, "u1:t1", set(), set(), gateway, 0, 0, full=True)

    value = output.getvalue()
    assert value.index("model stage=plan") < value.index("tool name=web_search")
    for fragment in ("input system", "选择工具", "input human", "去岘港", '"tool_name": "web_search"', '"query": "岘港景点"', "美溪海滩"):
        assert fragment in value


def test_full_trace_redacts_secret_fields_and_key_patterns() -> None:
    value = _trace_redact({"xsec_token": "private", "message": "token=sk-abcdefghijklmnop"})
    assert value["xsec_token"] == "[REDACTED]"
    assert "[REDACTED_KEY]" in value["message"]
