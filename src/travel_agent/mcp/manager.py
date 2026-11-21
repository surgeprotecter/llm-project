from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import os
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from travel_agent.mcp.config import MCPServerConfig


@dataclass
class MCPInvocation:
    success: bool
    data: dict[str, Any]
    summary: str = ""
    warnings: list[str] | None = None
    error_type: str | None = None
    error_message: str | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _run_async(coroutine):
    """Run an async MCP call from the synchronous ToolRegistry boundary."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # A caller may already be inside an async host. Keep the synchronous tool
    # protocol usable by running the MCP session in a short-lived worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


class MCPServerManager:
    def __init__(self, configs: list[MCPServerConfig]):
        self.configs = {config.name: config for config in configs if config.enabled}
        self.discovery_errors: dict[str, str] = {}
        self._workers: dict[str, _SessionWorker] = {}
        atexit.register(self.close)

    def _worker(self, config):
        worker = self._workers.get(config.name)
        if worker is None or worker.closed:
            worker = _SessionWorker(self, config)
            self._workers[config.name] = worker
        return worker

    def close(self):
        for worker in self._workers.values():
            worker.close()

    def cancel(self):
        for worker in self._workers.values():
            worker.cancel()

    def discover_tools(self) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []
        for config in self.configs.values():
            missing = _unresolved_env_keys(config)
            if missing:
                self.discovery_errors[config.name] = "missing environment variables: " + ", ".join(missing)
                continue
            try:
                tools = self._worker(config).request("list", None).tools
            except Exception as exc:  # noqa: BLE001 - isolate one unavailable MCP server
                self.discovery_errors[config.name] = _format_exception(exc, config)
                continue
            for tool in tools:
                remote_name = tool.name
                if config.allowed_tools and remote_name not in config.allowed_tools:
                    continue
                exposed_name = config.aliases.get(remote_name, f"{config.name}__{remote_name}")
                discovered.append(
                    {
                        "server_name": config.name,
                        "remote_name": remote_name,
                        "exposed_name": exposed_name,
                        "description": tool.description or remote_name,
                        "input_schema": tool.input_schema or {},
                        "output_schema": tool.output_schema or {},
                        "config": config,
                    }
                )
        return discovered

    def call_tool(self, server_name: str, remote_name: str, arguments: dict[str, Any]) -> MCPInvocation:
        config = self.configs[server_name]
        try:
            result = self._worker(config).request("call", (remote_name, arguments))
            data, summary = _normalise_result(result)
            if result.is_error or data.get("success") is False:
                return MCPInvocation(
                    False, data, summary=summary,
                    error_type=str(data.get("error_type") or "mcp_tool_error"),
                    error_message=str(data.get("error") or summary),
                )
            return MCPInvocation(True, data, summary=summary, warnings=[])
        except TimeoutError as exc:
            return MCPInvocation(False, {}, error_type="timeout", error_message=_format_exception(exc, config))
        except Exception as exc:  # noqa: BLE001 - convert external MCP failures
            return MCPInvocation(False, {}, error_type="mcp_error", error_message=_format_exception(exc, config))

    async def _discover_server(self, config: MCPServerConfig):
        async with self._session(config) as session:
            result = await asyncio.wait_for(session.list_tools(), timeout=config.timeout_seconds)
            return result.tools

    async def _call_server(self, config: MCPServerConfig, remote_name: str, arguments: dict[str, Any]) -> MCPInvocation:
        async with self._session(config) as session:
            result = await asyncio.wait_for(
                session.call_tool(remote_name, arguments), timeout=config.timeout_seconds
            )
        data, summary = _normalise_result(result)
        if result.is_error or data.get("success") is False:
            error = data.get("error")
            error_type = data.get("error_type")
            return MCPInvocation(
                False, data, summary=summary,
                error_type=error_type if isinstance(error_type, str) else "mcp_tool_error",
                error_message=error if isinstance(error, str) else summary,
            )
        return MCPInvocation(True, data, summary=summary, warnings=[])

    @asynccontextmanager
    async def _session(self, config: MCPServerConfig) -> AsyncIterator[ClientSession]:
        if config.transport == "stdio":
            parameters = StdioServerParameters(
                command=config.command or "",
                args=config.args,
                env=_environment(config.env),
            )
            with open(os.devnull, "w") as errlog:
                async with stdio_client(parameters, errlog=errlog) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await asyncio.wait_for(session.initialize(), timeout=config.timeout_seconds)
                        yield session
            return

        async with httpx.AsyncClient(headers=_headers(config), timeout=config.timeout_seconds) as client:
            async with streamable_http_client(_resolved_url(config), http_client=client) as streams:
                read_stream, write_stream = streams[:2]
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(session.initialize(), timeout=config.timeout_seconds)
                    yield session


class _SessionWorker:
    """Own one async session in one task for its whole lifetime.

    In-memory provider caches and stdio subprocess state survive tool calls.
    Enter/exit of anyio cancel scopes stays on the same task and event loop.
    """

    def __init__(self, manager, config):
        self.manager, self.config = manager, config
        self.ready = threading.Event()
        self.closed = False
        self.loop = None
        self.queue = None
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"mcp-{config.name}")
        self.thread.start()
        if not self.ready.wait(timeout=5):
            raise TimeoutError("MCP worker did not start")

    def _run(self):
        try:
            asyncio.run(self._serve())
        finally:
            self.closed = True
            self.ready.set()

    async def _serve(self):
        self.task = asyncio.current_task()
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()
        self.ready.set()
        pending = None
        try:
            async with self.manager._session(self.config) as session:
                while True:
                    item = await self.queue.get()
                    if item is None:
                        return
                    operation, arguments, pending = item
                    try:
                        call = (
                            session.list_tools() if operation == "list"
                            else session.call_tool(arguments[0], arguments[1])
                        )
                        result = await asyncio.wait_for(call, timeout=self.config.timeout_seconds)
                    except Exception as exc:
                        pending.set_exception(exc)
                    else:
                        pending.set_result(result)
                    pending = None
        except BaseException as exc:
            error = RuntimeError(_format_exception(exc, self.config))
            if pending and not pending.done():
                pending.set_exception(error)
            while not self.queue.empty():
                item = self.queue.get_nowait()
                if item and not item[2].done():
                    item[2].set_exception(error)
        finally:
            self.closed = True

    def request(self, operation, arguments):
        if self.closed:
            raise RuntimeError("MCP session is closed")
        future = concurrent.futures.Future()
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (operation, arguments, future))
        return future.result(timeout=self.config.timeout_seconds + 10)

    def close(self):
        if not self.closed and self.loop and not self.loop.is_closed():
            self.cancel()
            if threading.current_thread() is not self.thread:
                self.thread.join(timeout=0.5)

    def cancel(self):
        if not self.closed and self.loop and not self.loop.is_closed():
            task = getattr(self, "task", None)
            if task:
                self.loop.call_soon_threadsafe(task.cancel)


def _environment(configured: dict[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    for key, value in configured.items():
        environment[key] = os.path.expandvars(value)
    return environment


def _headers(config: MCPServerConfig) -> dict[str, str]:
    headers = {}
    for key, value in config.headers.items():
        missing = {name for name in _unresolved_variables(value) if not os.getenv(name)}
        if key in config.optional_headers and missing:
            continue
        headers[key] = os.path.expandvars(value)
    return headers


def _resolved_url(config: MCPServerConfig) -> str:
    return os.path.expandvars(config.url or "")


def _unresolved_env_keys(config: MCPServerConfig) -> list[str]:
    required_headers = [value for key, value in config.headers.items() if key not in config.optional_headers]
    values = [*config.env.values(), *required_headers, config.url or ""]
    names = {name for value in values for name in _unresolved_variables(value)}
    return sorted(name for name in names if not os.getenv(name))


def _unresolved_variables(value: str) -> set[str]:
    return set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value))


def _format_exception(exc: Exception, config: MCPServerConfig | None = None) -> str:
    if isinstance(exc, BaseExceptionGroup):
        details = [_format_exception(item, config) for item in exc.exceptions if isinstance(item, Exception)]
        return "; ".join(details) or type(exc).__name__
    message = str(exc).strip()
    if config:
        for value in [*config.env.values(), *config.headers.values(), config.url or ""]:
            for name in _unresolved_variables(value):
                secret = os.getenv(name)
                if secret:
                    message = message.replace(secret, "<redacted>")
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _normalise_result(result) -> tuple[dict[str, Any], str]:
    structured = result.structured_content
    if isinstance(structured, dict):
        data = structured
    elif structured is not None:
        data = {"result": structured}
    else:
        texts = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(str(text))
            else:
                dump = item.model_dump(mode="json") if hasattr(item, "model_dump") else str(item)
                texts.append(json.dumps(dump, ensure_ascii=False))
        combined = "\n".join(texts)
        try:
            parsed = json.loads(combined)
        except json.JSONDecodeError:
            parsed = combined
        data = parsed if isinstance(parsed, dict) else {"items": parsed}
    summary = data.get("summary") if isinstance(data.get("summary"), str) else json.dumps(data, ensure_ascii=False)[:2000]
    return data, summary
