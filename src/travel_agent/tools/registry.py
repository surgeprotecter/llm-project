from __future__ import annotations

import concurrent.futures
import copy
import time
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from travel_agent.models import ToolExecution, ToolResult
from travel_agent.persistence import Database
from travel_agent.tools.base import TravelTool


class ToolRegistry:
    def __init__(self, database: Database, max_retries: int = 2):
        self.database = database
        self.max_retries = max_retries
        self.tools: dict[str, TravelTool] = {}
        self.integration_errors: dict[str, str] = {}
        self.mcp_manager = None
        self.progress_callback = None
        self._legacy_imported = set()
        self._refresh = False
        self.cancelled = False
        self.parent = None

    def cancel(self):
        self.cancelled = True
        if self.mcp_manager:
            self.mcp_manager.cancel()

    def close(self):
        if self.mcp_manager:
            self.mcp_manager.close()

    def run_many(self, calls, thread_id):
        groups = {}
        for call in calls:
            tool = self.tools[call["name"]]
            server = getattr(tool, "server_name", call["name"].split("__")[0])
            groups.setdefault(server, []).append(call)
        if len(groups) < 2:
            return {c["id"]: self.run(c["name"], c["args"], thread_id) for c in calls}

        def run_group(group):
            # Each network worker owns its SQLite connection. Same MCP server is serial.
            database = Database(self.database.path)
            child = ToolRegistry(database, self.max_retries)
            child.tools = self.tools
            child.parent = self
            child.progress_callback = self.progress_callback
            child._legacy_imported.add(thread_id)
            try:
                return {c["id"]: child.run(c["name"], c["args"], thread_id) for c in group}
            finally:
                database.close()
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(groups))) as pool:
            futures = [pool.submit(run_group, group) for group in groups.values()]
            for future in concurrent.futures.as_completed(futures):
                result.update(future.result())
        return result

    def register(self, tool: TravelTool) -> None:
        if tool.spec.name in self.tools:
            raise ValueError(f"duplicate tool name: {tool.spec.name}")
        self.tools[tool.spec.name] = tool

    def has(self, name: str) -> bool:
        return name in self.tools

    def names_for_phase(self, phase: str) -> list[str]:
        return list(self.tools)

    def capability_catalog(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        selected = names or list(self.tools)
        return [
            {
                "name": name,
                # The planner only needs a routing hint.  The complete MCP/CLI
                # description and schema are loaded later via full_spec() after
                # a tool has been selected.
                "description": self._compact_description(self.tools[name].spec.description),
            }
            for name in selected
            if name in self.tools
        ]

    @staticmethod
    def _compact_description(description: str, limit: int = 360) -> str:
        normalized = " ".join(description.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit].rstrip() + "..."

    def full_spec(self, name: str) -> dict[str, Any] | None:
        tool = self.tools.get(name)
        if not tool:
            return None
        return tool.spec.model_dump(mode="json")

    def normalise_arguments(self, name, arguments):
        values = copy.deepcopy(arguments)
        tool = self.tools.get(name)
        for key, prop in (tool.spec.input_schema.get("properties", {}) if tool else {}).items():
            if key not in values and "default" in prop:
                values[key] = copy.deepcopy(prop["default"])
        return values

    def run(self, name: str, arguments: dict[str, Any], thread_id: str) -> ToolResult:
        if self.cancelled or (self.parent and self.parent.cancelled):
            from travel_agent.control import ExecutionCancelled
            raise ExecutionCancelled()
        if name not in self.tools:
            return ToolResult(success=False, tool_name=name, error_type="unknown_tool", error_message="工具不存在")
        tool = self.tools[name]
        arguments = self.normalise_arguments(name, arguments)
        if thread_id not in self._legacy_imported:
            self.database.runtime.migrate_executions(thread_id, self.normalise_arguments)
            self._legacy_imported.add(thread_id)
        try:
            if tool.spec.input_schema:
                Draft202012Validator(tool.spec.input_schema).validate(arguments)
        except ValidationError as exc:
            result = ToolResult(
                success=False,
                tool_name=name,
                error_type="invalid_arguments",
                error_message=f"工具参数不符合 Schema：{exc.message}",
                source_name=tool.spec.name,
                source_type=getattr(tool, "source_type", "mcp" if type(tool).__name__ == "MCPToolAdapter" else "mock"),
            )
            self.database.save_tool_execution(ToolExecution(thread_id=thread_id, tool_name=name, arguments=arguments, result=result, attempt=1))
            return result
        if name in {"china_rail__get-tickets", "china_rail__get-interline-tickets"}:
            try:
                travel_date = date.fromisoformat(arguments.get("date", ""))
            except (TypeError, ValueError):
                travel_date = None
            today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
            if travel_date and travel_date > today + timedelta(days=14):
                opens = (travel_date - timedelta(days=14)).isoformat()
                result = ToolResult(
                    success=False, tool_name=name, source_type="internal", source_name="12306 预售校验",
                    error_type="outside_booking_window",
                    error_message=f"所选日期超出12306通常的15天预售期，预计{opens}起分时开售。保留用户日期，届时再查实时票。",
                    source_url="https://help.12306.cn/cms/answer/h5/001531826595438842083440b580b2e0",
                    data={"travel_date": travel_date.isoformat(), "expected_sale_date": opens},
                )
                self.database.save_tool_execution(ToolExecution(thread_id=thread_id, tool_name=name, arguments=arguments, result=result, attempt=1))
                return result
        cached = None if self._refresh else self.database.runtime.cached(thread_id, name, arguments)
        if cached:
            self.database.save_tool_execution(ToolExecution(
                thread_id=thread_id, tool_name=name, arguments=arguments, result=cached,
            ))
            if self.progress_callback:
                self.progress_callback("tool", {"name": name, "reused": True, "result_id": cached.result_id})
            return cached
        if self.progress_callback:
            self.progress_callback("tool", {"name": name, "status": "running"})
        started = time.perf_counter()
        last_result: ToolResult | None = None
        retries = getattr(tool, "max_retries", self.max_retries)
        for attempt in range(1, retries + 2):
            if self.cancelled or (self.parent and self.parent.cancelled):
                from travel_agent.control import ExecutionCancelled
                raise ExecutionCancelled()
            try:
                result = tool.run(arguments)
                last_result = result
                if result.success or attempt > retries:
                    break
            except Exception as exc:  # noqa: BLE001 - tool boundary must convert failures
                last_result = ToolResult(
                    success=False,
                    tool_name=name,
                    error_type="tool_exception",
                    error_message=str(exc),
                    source_name=tool.spec.name,
                    source_type=getattr(tool, "source_type", "mcp" if type(tool).__name__ == "MCPToolAdapter" else "mock"),
                )
                if attempt > retries:
                    break
        assert last_result is not None
        duration_ms = int((time.perf_counter() - started) * 1000)
        self.database.save_tool_execution(
            ToolExecution(
                thread_id=thread_id,
                tool_name=name,
                arguments=arguments,
                result=last_result,
                duration_ms=duration_ms,
                attempt=attempt,
            )
        )
        if self.progress_callback:
            self.progress_callback("tool", {"name": name, "success": last_result.success, "result_id": last_result.result_id})
        return last_result
