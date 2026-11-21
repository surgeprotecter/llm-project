"""Bounded, keyless Exa MCP search behind the existing ``web_search`` alias."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from travel_agent.models import ToolResult, ToolSpec
from travel_agent.tools.base import TravelTool

_EXA_URL = "https://mcp.exa.ai/mcp"
_TIMEOUT = 25.0
_MCP_TIMEOUT = 10.0
_SOURCE = "exa_mcp"
_DEFAULT_OBJECTIVE = (
    "Find relevant source pages answering the query, prioritizing official operators and "
    "government sources. Return source URLs and source text with requested facts and figures."
)


class _Text(HTMLParser):
    def __init__(self, value: str):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.feed(value)

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _text(value: Any) -> str:
    return " ".join(" ".join(_Text(str(value or "")).parts).split())


def _plaintext_results(value: str) -> list[dict[str, str]]:
    """Exa's MCP server returns source text, rather than JSON, in text blocks."""
    rows = []
    for match in re.finditer(
        r"^Title:[ \t]*([^\r\n]+)\r?\nURL:[ \t]*(https?://[^\s]+)[ \t]*\r?\n"
        r"(.*?)(?=^Title:[ \t]*|\Z)",
        value,
        re.MULTILINE | re.DOTALL,
    ):
        title, url, rest = match.groups()
        content = re.search(r"^(Text|Highlights|Content):[ \t]*", rest, re.MULTILINE)
        if not content:
            continue
        body = re.sub(r"\n\s*---\s*$", "", rest[content.end() :]).strip()
        if body:
            rows.append(
                {
                    "title": title,
                    "url": url,
                    "body": body,
                    "snippet_source": f"exa_{content.group(1).lower()}",
                }
            )
    return rows


def _results(data: Any, limit: int) -> list[dict[str, str]]:
    """Normalize structured results or MCP Title/URL/Text/Highlights blocks."""
    pending = [data]
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    while pending and len(found) < limit:
        item = pending.pop(0)
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except ValueError:
                pending.extend(_plaintext_results(item))
            else:
                if isinstance(parsed, (dict, list)):
                    pending.append(parsed)
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("href") or item.get("link") or "").strip()
            try:
                parsed_url = urlsplit(url)
            except ValueError:
                continue
            title = _text(item.get("title"))
            body = item.get("text") or item.get("body") or item.get("highlights")
            body = body or item.get("description") or item.get("snippet") or ""
            if isinstance(body, list):
                body = "\n".join(str(part) for part in body)
            description = _text(body)
            if (
                parsed_url.scheme in {"http", "https"}
                and parsed_url.hostname
                and title
                and description
            ):
                if url not in seen:
                    seen.add(url)
                    found.append(
                        {
                            "title": title,
                            "url": url,
                            "description": description[:1800],
                            "body": str(body)[:12000],
                            "engine": _SOURCE,
                            "source_url": url,
                            "search_url": _EXA_URL,
                            "snippet_source": str(item.get("snippet_source") or "exa_source_text"),
                        }
                    )
            else:
                for key in ("data", "web", "results", "items", "content", "result"):
                    if key in item:
                        pending.append(item[key])
                if isinstance(item.get("text"), str):
                    pending.append(item["text"])
    return found


def _search_worker() -> None:
    """Use the real MCP manager; the parent bounds discovery, call and shutdown."""
    from travel_agent.mcp.config import MCPServerConfig
    from travel_agent.mcp.manager import MCPServerManager

    request = json.load(sys.stdin)
    config = MCPServerConfig(
        name="exa",
        transport="streamable_http",
        url=_EXA_URL,
        allowed_tools=["web_search_exa"],
        aliases={"web_search_exa": "web_search"},
        timeout_seconds=_MCP_TIMEOUT,
        max_retries=0,
    )
    manager = MCPServerManager([config])
    started = time.monotonic()
    discovered = manager.discover_tools()
    metadata = next((tool for tool in discovered if tool["remote_name"] == "web_search_exa"), None)
    diagnostic: dict[str, Any] = {"engine": _SOURCE, "endpoint": _EXA_URL}
    if metadata is None:
        diagnostic["reason"] = manager.discovery_errors.get("exa", "web_search_exa not discovered")
        print(
            json.dumps(
                {
                    "success": False,
                    "results": [],
                    "diagnostics": [diagnostic],
                    "error_type": "mcp_discovery_error",
                    "error_message": diagnostic["reason"],
                }
            )
        )
        return
    diagnostic["input_schema"] = metadata["input_schema"]
    invocation = manager.call_tool("exa", "web_search_exa", request["arguments"])
    rows = _results(invocation.data, request["limit"]) if invocation.success else []
    diagnostic.update(
        elapsed_seconds=round(time.monotonic() - started, 3),
        accepted_results=len(rows),
        remote_tool="web_search_exa",
    )
    error = invocation.error_message or (
        "Exa returned no usable source URLs and text" if not rows else None
    )
    if error:
        diagnostic["reason"] = error
        diagnostic["response_excerpt"] = invocation.summary[:1200]
    print(
        json.dumps(
            {
                "success": bool(rows),
                "results": rows,
                "diagnostics": [diagnostic],
                "error_type": invocation.error_type or ("empty_search" if not rows else None),
                "error_message": error,
                "warnings": invocation.warnings or [],
            }
        )
    )


class FreeWebSearchTool(TravelTool):
    """Expose official keyless Exa MCP under the builder's existing alias.

    A discovered Exa adapter can be supplied by MCP configuration. Legacy
    Firecrawl adapters are bypassed; otherwise discovery uses the official URL.
    No HTML scraping, model calls, or automatic retries are performed.
    """

    source_type = "mcp"
    max_retries = 0

    def __init__(self, primary: TravelTool | None = None):
        self.primary = (
            primary if getattr(primary, "remote_name", None) == "web_search_exa" else None
        )
        self.spec = ToolSpec(
            name="web_search",
            description="Search the web using keyless Exa MCP; return real source URLs and text.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "objective": {"type": "string", "maxLength": 4096},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    "count": {"type": "integer", "minimum": 1, "maximum": 10},
                    "numResults": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        try:
            limit = min(
                10,
                max(
                    1,
                    int(
                        arguments.get(
                            "limit", arguments.get("numResults", arguments.get("count", 5))
                        )
                    ),
                ),
            )
        except (TypeError, ValueError, OverflowError):
            limit = 5
        if not query:
            return ToolResult(
                success=False,
                tool_name="web_search",
                source_type="mcp",
                source_name=_SOURCE,
                error_type="invalid_arguments",
                error_message="query must not be empty",
            )
        remote_arguments = {
            "query": query,
            "numResults": limit,
            "objective": str(arguments.get("objective") or _DEFAULT_OBJECTIVE).strip()
            or _DEFAULT_OBJECTIVE,
        }
        if len(remote_arguments["objective"]) > 4096:
            return ToolResult(
                success=False,
                tool_name="web_search",
                source_type="mcp",
                source_name=_SOURCE,
                error_type="invalid_arguments",
                error_message="objective must be at most 4096 characters",
            )
        try:
            if self.primary is not None:
                completed: queue.Queue[ToolResult | Exception] = queue.Queue(maxsize=1)

                def invoke() -> None:
                    try:
                        completed.put(self.primary.run(remote_arguments))
                    except Exception as exc:
                        completed.put(exc)

                threading.Thread(target=invoke, daemon=True, name="web-search-exa").start()
                result = completed.get(timeout=_TIMEOUT)
                if isinstance(result, Exception):
                    raise result
                rows = _results(result.data, limit) if result.success else []
                payload = {
                    "results": rows,
                    "warnings": result.warnings,
                    "diagnostics": [
                        {
                            "engine": _SOURCE,
                            "endpoint": _EXA_URL,
                            "remote_tool": "web_search_exa",
                            "accepted_results": len(rows),
                        }
                    ],
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                }
            else:
                worker = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from travel_agent.tools.web_search import _search_worker; _search_worker()",
                    ],
                    input=json.dumps({"arguments": remote_arguments, "limit": limit}),
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=_TIMEOUT,
                )
                payload = json.loads(worker.stdout)
                rows = payload["results"]
        except Exception as exc:  # External MCP/process boundary: preserve the real failure.
            detail = str(exc)[:400] or type(exc).__name__
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                detail = exc.stderr.strip()[-600:]
            return ToolResult(
                success=False,
                tool_name="web_search",
                source_type="mcp",
                source_name=_SOURCE,
                data={
                    "query": query,
                    "results": [],
                    "engines_attempted": [_SOURCE],
                    "endpoint": _EXA_URL,
                },
                error_type="timeout"
                if isinstance(exc, (subprocess.TimeoutExpired, queue.Empty))
                else "search_unavailable",
                error_message=f"Exa MCP failed ({type(exc).__name__}): {detail}; no results invented.",
            )
        now = datetime.now(UTC)
        return ToolResult(
            success=bool(rows),
            tool_name="web_search",
            data={
                "query": query,
                "results": rows,
                "engines_attempted": [_SOURCE],
                "diagnostics": payload.get("diagnostics", []),
                "endpoint": _EXA_URL,
            },
            summary=f"Retrieved {len(rows)} Exa source results."
            if rows
            else "Exa returned no usable results.",
            source_type="mcp",
            source_name=_SOURCE,
            source_url=rows[0]["url"] if rows else None,
            fetched_at=now,
            valid_until=now + timedelta(minutes=30) if rows else None,
            warnings=payload.get("warnings", []),
            error_type=None if rows else payload.get("error_type") or "empty_search",
            error_message=None
            if rows
            else payload.get("error_message")
            or "Exa returned no usable source URLs and text; no results invented.",
        )
