from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import typer
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from travel_agent.config import Settings
from travel_agent.graph import build_graph
from travel_agent.mcp import MCPServerManager, load_mcp_configs
from travel_agent.persistence import Database

app = typer.Typer(help="本地旅行规划 Agent")
console = Console()


class ChatInput:
    """Keep pasted Unicode and newlines intact, including in redirected stdin."""

    def __init__(self, multiline: bool = False):
        self.multiline = multiline
        self.source = "stdin"
        self.session = None
        if sys.stdin.isatty():
            try:
                from prompt_toolkit import PromptSession
                from prompt_toolkit.key_binding import KeyBindings

                bindings = KeyBindings()

                @bindings.add("escape", "enter")
                def submit(event):
                    event.current_buffer.validate_and_handle()

                @bindings.add("f2")
                def toggle_multiline(event):
                    self.multiline = not self.multiline

                self.session = PromptSession(key_bindings=bindings, enable_open_in_editor=True)
            except ImportError:
                pass

    def _read_line(self, prompt: str, multiline: bool | None = None) -> str:
        if self.session is not None:
            from prompt_toolkit.filters import Condition

            return self.session.prompt(prompt, multiline=Condition(lambda: self.multiline) if multiline is None else multiline)
        console.print(prompt, end="", markup=False)
        value = sys.stdin.readline()
        if value == "":
            raise EOFError
        return value.removesuffix("\n").removesuffix("\r")

    def read(self) -> str:
        while True:
            self.source = "terminal" if self.session is not None else "stdin"
            value = self._read_line("你：")
            command = value.strip()
            if command == "/multiline":
                self.multiline = not self.multiline
                console.print("多行模式：" + ("开启；Esc+Enter 提交。" if self.multiline else "关闭。"))
                continue
            if command == "/input":
                console.print("逐行输入或粘贴，单独一行 /send 提交，/cancel 取消。")
                lines = []
                while True:
                    line = self._read_line("… ", multiline=False)
                    if line.strip() == "/cancel":
                        break
                    if line.strip() == "/send":
                        self.source = "multiline"
                        return "\n".join(lines)
                    lines.append(line)
                continue
            if command.startswith("/input "):
                path = command[len("/input "):].strip().strip("\"'")
                try:
                    value = Path(path).expanduser().read_bytes().decode("utf-8")
                except (OSError, UnicodeError) as exc:
                    console.print(f"无法读取输入文件：{exc}", markup=False)
                    continue
                self.source = "file"
            return value


def _record_input(database: Database, thread_id: str, user_id: str, trip_id: str, raw: str, source: str, trace: bool) -> None:
    payload = {"raw_input": raw, "source": source, "characters": len(raw), "lines": raw.count("\n") + 1}
    database.append_event(thread_id, user_id, trip_id, "user.input_received", payload)
    if trace:
        preview = _trace_redact(raw)
        if len(preview) > 180:
            preview = preview[:180] + "…"
        console.print(
            f"[trace] input source={source} chars={len(raw)} lines={payload['lines']} "
            f"preview={json.dumps(preview, ensure_ascii=False)}",
            markup=False,
        )


def _settings(database_path: str | None = None, require_real_model: bool = False) -> Settings:
    settings = Settings.from_env()
    if database_path:
        settings = replace(settings, database_path=Path(database_path))
    if require_real_model:
        settings = replace(settings, require_real_model=True, tool_mode="real")
    return settings


def _interrupt_question(result: dict[str, Any]) -> str | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    if isinstance(value, dict):
        return str(value.get("question", "请补充信息。"))
    return str(value)


def _friendly_execution_error(error: Exception) -> str:
    message = str(error)
    if "Insufficient Balance" in message or "Error code: 402" in message:
        return "DeepSeek API 余额不足（402）。本轮未完成，已有报告和会话已保留；请充值或更换有效 Key 后重试。"
    if "timed out" in message.lower() or "timeout" in message.lower():
        return "模型服务响应超时。本轮没有完成，请稍后重试；已保留之前的会话状态。"
    return f"本轮执行失败：{message[:300]}"


def _print_trace(
    database: Database,
    thread_id: str,
    previous_execution_ids: set[str],
    previous_event_ids: set[int],
    gateway,
    previous_model_calls: int,
    previous_model_attempts: int,
    full: bool = False,
) -> None:
    executions = [
        item
        for item in database.list_tool_executions(thread_id)
        if item["execution_id"] not in previous_execution_ids
    ]
    for item in executions:
        payload = item["payload"]
        result = payload.get("result", {})
        status = "success" if result.get("success") else f"error:{result.get('error_type', 'unknown')}"
        source = "/".join(filter(None, [result.get("source_type"), result.get("source_name")])) or "unknown"
        console.print(
            f"[dim][trace] tool={item['tool_name']} source={source} status={status} "
            f"duration={payload.get('duration_ms', 0)}ms attempt={payload.get('attempt', 1)}[/dim]"
        )

    for event in database.list_events(thread_id):
        if event["event_id"] in previous_event_ids:
            continue
        payload = event["payload"]
        if event["event_type"] == "agent.plan_decision":
            console.print(
                f"[dim][trace] plan action={payload.get('action')} tool={payload.get('tool_name') or '-'} "
                f"spec_loaded={payload.get('loaded_tool_spec', False)} round={payload.get('planner_round')}[/dim]"
            )
        elif event["event_type"] == "agent.result_assessed":
            console.print(
                f"[dim][trace] assessment sufficient={payload.get('sufficient')} "
                f"next={payload.get('next_action')} tools={payload.get('tool_count')}[/dim]"
            )
        elif event["event_type"] == "agent.request_scope":
            console.print(f"[trace] scope={payload['scenario']} artifact={payload['artifact']}", markup=False, style="dim")
        elif event["event_type"] == "agent.delivery_decision":
            console.print(f"[trace] delivery={payload['decision']} scenario={payload['scenario']}", markup=False, style="dim")

    for attempt in gateway.attempt_trace[previous_model_attempts:]:
        console.print(
            f"[dim][trace] model_stage={attempt['stage']} attempt={attempt['attempt']} "
            f"duration={attempt['duration_ms']}ms prompt_chars={attempt['prompt_chars']} "
            f"input_tokens={attempt.get('usage',{}).get('input_tokens','unavailable')} "
            f"output_tokens={attempt.get('usage',{}).get('output_tokens','unavailable')} "
            f"status={attempt['status']}[/dim]"
        )

    snapshot = gateway.trace_snapshot()
    model_calls = snapshot["call_count"] - previous_model_calls
    if model_calls:
        status = snapshot["last_status"]
        console.print(
            f"[dim][trace] model={gateway.settings.model_name} calls={model_calls} "
            f"status={status} success_total={snapshot['success_count']}[/dim]"
        )

    manifest = database.latest_context_manifest(thread_id)
    if manifest:
        context_tokens = manifest.get("total_tokens_estimate") or (
            sum(manifest.get("section_tokens", {}).values()) + manifest.get("reserved_tool_result_tokens", 0)
        )
        console.print(
            f"[dim][trace] context phase={manifest.get('phase')} "
            f"memory={len(manifest.get('selected_memory_ids', []))} "
            f"evidence={len(manifest.get('selected_evidence_ids', []))} "
            f"tokens≈{context_tokens} compacted={manifest.get('compacted', False)}[/dim]"
        )

    if full:
        timeline: list[tuple[str, str, dict[str, Any]]] = [
            (item["created_at"], "tool", item["payload"]) for item in executions
        ]
        timeline.extend(
            (item["started_at"], "model", item)
            for item in gateway.attempt_trace[previous_model_attempts:]
        )
        for _, kind, payload in sorted(timeline, key=lambda entry: entry[0]):
            if kind == "model":
                console.print(
                    f"[trace-full] model stage={payload['stage']} attempt={payload['attempt']} "
                    f"status={payload['status']}",
                    markup=False,
                )
                for message in payload.get("input_messages", []):
                    console.print(f"  input {message['role']}:\n{_trace_redact(message['content'])}", markup=False)
                    if message.get("tool_calls"):
                        console.print(f"  input tool_calls:\n{_trace_json(_trace_redact(message['tool_calls']))}", markup=False)
                if payload.get("input_tools"):
                    console.print(f"  input tool schemas:\n{_trace_json(_trace_redact(payload['input_tools']))}", markup=False)
                if "output" in payload:
                    console.print(f"  output:\n{_trace_json(_trace_redact(payload['output']))}", markup=False)
                if payload.get("output_tool_calls"):
                    console.print(f"  output tool_calls:\n{_trace_json(_trace_redact(payload['output_tool_calls']))}", markup=False)
            else:
                result = payload.get("result", {})
                console.print(
                    f"[trace-full] tool name={payload.get('tool_name')} "
                    f"source={result.get('source_type')}/{result.get('source_name')} "
                    f"success={result.get('success')}",
                    markup=False,
                )
                console.print(f"  arguments:\n{_trace_json(_trace_redact(payload.get('arguments', {})))}", markup=False)
                console.print(f"  result:\n{_trace_json(_trace_redact(result))}", markup=False)


def _trace_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _trace_redact(value: Any, key: str = "") -> Any:
    usage_keys={"input_tokens","output_tokens","total_tokens","completion_tokens","prompt_tokens","prompt_cache_hit_tokens","prompt_cache_miss_tokens","cached_tokens","cache_write_tokens","audio_tokens","image_tokens","text_tokens","input_token_details","output_token_details","completion_tokens_details","prompt_tokens_details","reasoning_tokens","accepted_prediction_tokens","rejected_prediction_tokens","max_tokens","reserved_tool_result_tokens","context_token_budget"}
    sensitive = key.lower() not in usage_keys and any(token in key.lower() for token in ("api_key", "apikey", "authorization", "password", "cookie", "token", "secret"))
    if sensitive:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {name: _trace_redact(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_trace_redact(item, key) for item in value]
    if isinstance(value, str):
        try:
            embedded = json.loads(value)
        except (TypeError, ValueError):
            embedded = None
        if isinstance(embedded, (dict, list)):
            return json.dumps(_trace_redact(embedded), ensure_ascii=False)
        return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED_KEY]", value)
    return value


@app.command()
def chat(
    user_id: str = typer.Option("default", "--user-id"),
    trip_id: str = typer.Option("default-trip", "--trip-id"),
    database_path: str | None = typer.Option(None, "--database-path"),
    trace: bool = typer.Option(False, "--trace", help="显示简要执行轨迹，不打印密钥或完整网页内容。"),
    trace_full: bool = typer.Option(False, "--trace-full", help="显示每次模型输入/输出和工具参数/完整结果；可能含个人信息。"),
    real_only: bool = typer.Option(False, "--real-only", help="只运行真实模型与已连接的真实工具；绝不注册 Mock 旅行工具。"),
    input_file: Path | None = typer.Option(None, "--input-file", "--input", exists=True, dir_okay=False, readable=True, help="将 UTF-8 文件完整内容作为首条消息。"),
    multiline: bool = typer.Option(False, "--multiline", help="多行编辑，Esc+Enter 提交；F2 切换单行/多行。"),
) -> None:
    """启动一个可恢复的旅行规划会话。"""
    try:
        first_input = input_file.expanduser().read_bytes().decode("utf-8") if input_file else None
    except (OSError, UnicodeError) as exc:
        raise typer.BadParameter(f"无法读取 UTF-8 输入文件：{exc}", param_hint="--input-file") from exc
    settings = _settings(database_path, require_real_model=real_only)
    if settings.strict_real and not settings.api_key:
        raise typer.BadParameter("真实模式要求配置 DEEPSEEK_API_KEY。")
    database = Database(settings.database_path)
    graph, registry, gateway = build_graph(settings, database)
    gateway.capture_io = trace_full
    def live_progress(kind, payload):
        if kind == "plan":
            for task in payload["tasks"]:
                marker = {"completed": "✓", "in_progress": "→", "pending": "·"}[task["status"]]
                console.print(f"[dim]{marker} {task['description']}[/dim]", markup=True)
        elif kind == "action":
            console.print(f"[dim]执行 {payload['name']} {payload.get('operation') or ''}[/dim]")
        elif kind == "tool":
            detail = "复用已有结果" if payload.get("reused") else payload.get("status", "完成" if payload.get("success") else "未成功")
            console.print(f"[dim]工具 {payload['name']} · {detail}[/dim]")
    gateway.progress_callback = live_progress
    trace = trace or trace_full
    if settings.strict_real and not gateway.enabled:
        database.close()
        raise typer.BadParameter("真实模式要求已配置且可初始化的 DeepSeek API。")
    if settings.strict_real and not any(type(tool).__name__ == "MCPToolAdapter" for tool in registry.tools.values()):
        database.close()
        details = "; ".join(f"{name}: {error}" for name, error in registry.integration_errors.items())
        raise typer.BadParameter("真实模式没有任何 MCP 工具成功连接；不会退回 Mock。" + (f" {details}" if details else ""))
    thread_id = f"{user_id}:{trip_id}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    console.print("[bold cyan]旅行规划 Agent[/bold cyan]")
    if trace:
        model_mode = "real API enabled" if gateway.enabled else "local fallback"
        console.print(f"[dim][trace] model={settings.model_name} status={model_mode}[/dim]")
        if gateway.enabled:
            console.print(
                f"[dim][trace] model_config thinking={gateway.thinking_mode} "
                f"base_output_limit={gateway.max_output_tokens} agent_output_limit=2200 "
                f"context_window={settings.context_token_budget} summary_model={settings.summary_model_name}[/dim]"
            )
        mcp_tools = [name for name, tool in registry.tools.items() if type(tool).__name__ == "MCPToolAdapter"]
        console.print(f"[dim][trace] mcp_tools={mcp_tools or 'none'}[/dim]")
        other_real = [name for name,tool in registry.tools.items() if getattr(tool,"source_type",None) in {"api","cli"}]
        if other_real:
            console.print(f"[dim][trace] real_api_cli_tools={other_real}[/dim]")
        console.print(f"[dim][trace] tool_mode={settings.tool_mode} mock_tools_registered={settings.tool_mode == 'mock'}[/dim]")
    console.print("输入 quit 退出；/input 路径 读取 UTF-8 文件；/input 后逐行粘贴并用 /send 提交。")
    console.print("多行编辑：--multiline 或 F2 切换，Esc+Enter 提交；Ctrl-X Ctrl-E 打开外部编辑器。\n")
    if registry.integration_errors:
        for server_name, error in registry.integration_errors.items():
            console.print(f"[yellow]MCP Server {server_name} 未连接：{error}[/yellow]")
    reader = ChatInput(multiline=multiline)
    try:
        while True:
            try:
                if first_input is not None:
                    user_input, first_input = first_input, None
                    source = "file"
                else:
                    user_input = reader.read()
                    source = reader.source
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print("\n已取消当前输入。")
                continue
            if user_input.strip().lower() in {"quit", "exit", "退出"}:
                break
            if not user_input.strip():
                continue
            _record_input(database, thread_id, user_id, trip_id, user_input, source, trace)
            previous_execution_ids = {item["execution_id"] for item in database.list_tool_executions(thread_id)}
            previous_event_ids = {item["event_id"] for item in database.list_events(thread_id)}
            previous_model_calls = gateway.call_count
            previous_model_attempts = len(gateway.attempt_trace)
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "user_id": user_id,
                "trip_id": trip_id,
                "thread_id": thread_id,
                "current_query": user_input,
            }
            try:
                result = graph.invoke(initial_state, config)
                while (question := _interrupt_question(result)) is not None:
                    console.print(f"[yellow]Agent：{question}[/yellow]")
                    answer = reader.read()
                    if answer.strip().lower() in {"quit", "exit", "退出"}:
                        return
                    _record_input(database, thread_id, user_id, trip_id, answer, reader.source, trace)
                    result = graph.invoke(Command(resume=answer), config)
            except (EOFError, KeyboardInterrupt):
                console.print("\n会话已保留。")
                break
            except Exception as exc:  # noqa: BLE001 - CLI should keep the session usable after provider errors
                console.print(f"[red]{_friendly_execution_error(exc)}[/red]")
                if trace:
                    _print_trace(database, thread_id, previous_execution_ids, previous_event_ids, gateway, previous_model_calls, previous_model_attempts, full=trace_full)
                continue
            response = result.get("assistant_response")
            if response:
                console.print(Markdown(str(response)))
            else:
                console.print("[yellow]本轮没有生成最终回复，请继续补充。[/yellow]")
            if trace:
                _print_trace(database, thread_id, previous_execution_ids, previous_event_ids, gateway, previous_model_calls, previous_model_attempts, full=trace_full)
    finally:
        gateway.close()
        registry.close()
        database.close()


@app.command("export-report")
def export_report(
    trip_id: str = typer.Option("default-trip", "--trip-id"),
    database_path: str | None = typer.Option(None, "--database-path"),
) -> None:
    """查看指定旅行最近一次生成的报告路径。"""
    settings = _settings(database_path)
    database = Database(settings.database_path)
    report = database.latest_report(trip_id)
    if not report:
        console.print(f"[yellow]没有找到旅行 {trip_id} 的报告。[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"Markdown：{report['markdown_path']}")
    console.print(f"HTML：{report['html_path']}")
    database.close()


@app.command("check-tools")
def check_tools() -> None:
    """连接 MCP 并发现工具；不代表已经验证具体查询结果，不输出凭证。"""
    settings = Settings.from_env()
    configs = load_mcp_configs(settings.mcp_config_path)
    manager = MCPServerManager(configs)
    discovered = manager.discover_tools()
    by_server: dict[str, list[str]] = {}
    for item in discovered:
        by_server.setdefault(item["server_name"], []).append(item["exposed_name"])
    console.print(f"模型：{settings.model_name}；API Key：{'已配置' if settings.api_key else '未配置'}")
    console.print(f"工具模式：{settings.tool_mode}；MCP Server 配置数：{len(configs)}")
    for config in configs:
        if config.name in manager.discovery_errors:
            console.print(f"[red]不可用[/red] {config.name}: {_trace_redact(manager.discovery_errors[config.name])}")
        else:
            names = by_server.get(config.name, [])
            console.print(f"[green]已连接[/green] {config.name}: {', '.join(names) or '未发现允许列表内工具'}")
    if not configs:
        console.print("[yellow]没有 MCP Server 配置。[/yellow]")
    if not discovered:
        raise typer.Exit(code=1)


@app.command("inspect-context")
def inspect_context(
    trip_id: str = typer.Option("default-trip", "--trip-id"),
    user_id: str = typer.Option("default", "--user-id"),
    database_path: str | None = typer.Option(None, "--database-path"),
) -> None:
    """查看最近一次模型上下文选择清单。"""
    settings = _settings(database_path)
    database = Database(settings.database_path)
    manifest = database.latest_context_manifest(f"{user_id}:{trip_id}")
    if manifest is None:
        console.print("[yellow]还没有上下文清单。[/yellow]")
    else:
        console.print_json(json.dumps(manifest, ensure_ascii=False))
    database.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
