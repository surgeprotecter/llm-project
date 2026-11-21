from __future__ import annotations

import json
import subprocess
from typing import Any

import httpx

from travel_agent.models import ToolResult, ToolSpec
from travel_agent.tools.base import TravelTool


class CliToolAdapter(TravelTool):
    """Safe adapter for a preconfigured CLI command.

    The command is fixed by application configuration; model output is used only as JSON arguments.
    """

    def __init__(self, spec: ToolSpec, command: list[str], timeout_seconds: int = 30):
        self.spec = spec
        self.command = command
        self.timeout_seconds = timeout_seconds

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(arguments, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, tool_name=self.spec.name, error_type="timeout", error_message="CLI 工具超时", source_type="cli", source_name=self.spec.name)
        if completed.returncode != 0:
            return ToolResult(success=False, tool_name=self.spec.name, error_type="cli_error", error_message=completed.stderr[-1000:], source_type="cli", source_name=self.spec.name)
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError:
            data = {"stdout": completed.stdout}
        return ToolResult(success=True, tool_name=self.spec.name, data=data, summary="CLI 工具执行完成。", source_type="cli", source_name=self.spec.name)


class HttpApiToolAdapter(TravelTool):
    """Adapter for a fixed HTTP endpoint; model output becomes validated query parameters."""

    def __init__(self, spec: ToolSpec, url: str, headers: dict[str, str] | None = None, timeout_seconds: int = 30):
        self.spec = spec
        self.url = url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            response = httpx.get(self.url, params=arguments, headers=self.headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException:
            return ToolResult(success=False, tool_name=self.spec.name, error_type="timeout", error_message="HTTP 工具超时", source_type="api", source_name=self.spec.name)
        except (httpx.HTTPError, ValueError) as exc:
            return ToolResult(success=False, tool_name=self.spec.name, error_type="api_error", error_message=str(exc), source_type="api", source_name=self.spec.name)
        return ToolResult(success=True, tool_name=self.spec.name, data=data if isinstance(data, dict) else {"items": data}, summary="HTTP 工具执行完成。", source_type="api", source_name=self.spec.name, source_url=self.url)


class SkillAdapter(TravelTool):
    """Adapter for an already-loaded Skill callable.

    Skill loading and authentication remain outside the graph. The graph only sees this protocol.
    """

    def __init__(self, spec: ToolSpec, handler):
        self.spec = spec
        self.handler = handler

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            output = self.handler(arguments)
        except Exception as exc:  # noqa: BLE001 - isolate external Skill failures
            return ToolResult(success=False, tool_name=self.spec.name, error_type="skill_error", error_message=str(exc), source_type="skill", source_name=self.spec.name)
        data = output if isinstance(output, dict) else {"result": output}
        return ToolResult(success=True, tool_name=self.spec.name, data=data, summary="Skill 执行完成。", source_type="skill", source_name=self.spec.name)
