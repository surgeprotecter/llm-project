from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from travel_agent.config import Settings


class ModelGateway:
    """Optional DeepSeek gateway with a deterministic fallback for offline runs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.call_count = 0
        self.success_count = 0
        self.fallback_count = 0
        self.last_status = "not_called"
        self.last_error: str | None = None
        self.attempt_trace: list[dict[str, Any]] = []
        self.capture_io = False
        self.usage_recorder = None
        self.turn_usage_start = 0
        self.progress_callback = None
        self.worker = None
        self.thinking_mode = "disabled"
        self.max_output_tokens = 1024
        if settings.api_key:
            try:
                from langchain_openai import ChatOpenAI

                self.model = ChatOpenAI(
                    api_key=settings.api_key,
                    base_url=settings.api_base,
                    model=settings.model_name,
                    temperature=0.2,
                    timeout=45,
                    # DeepSeek V4 Pro enables high-effort thinking by default.
                    # These short JSON/summary tasks do not need it, and the
                    # default output allowance is otherwise very large.
                    extra_body={"thinking": {"type": self.thinking_mode}},
                    # Retries are bounded in _invoke_with_retry so the trace
                    # reflects actual provider requests and we do not stack
                    # LangChain retries with application retries.
                    max_retries=0,
                )
            except Exception:
                self.model = None

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def generate(
        self, system_prompt: str, user_prompt: str, fallback: str, *, stage: str = "generate"
    ) -> str:
        self.last_error = None
        if not self.model:
            if self.settings.strict_real:
                self.last_status = "error_no_model"
                raise RuntimeError("真实模型未启用：请检查 DEEPSEEK_API_KEY 和模型配置。")
            self.fallback_count += 1
            self.last_status = "fallback_no_model"
            return fallback
        try:
            response = self._invoke_with_retry(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)], stage
            )
            self.success_count += 1
            self.last_status = "success"
            return self._content_text(response)
        except Exception as exc:  # noqa: BLE001 - provider errors should not break demo mode
            if self.settings.strict_real:
                self.last_status = "error_provider"
                self.last_error = str(exc)
                raise RuntimeError(f"真实模型调用失败：{exc}") from exc
            self.fallback_count += 1
            self.last_status = "fallback_provider_error"
            self.last_error = str(exc)
            return f"{fallback}\n\n（模型服务暂时不可用，已使用本地结果：{exc}）"

    def trace_snapshot(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "success_count": self.success_count,
            "fallback_count": self.fallback_count,
            "last_status": self.last_status,
            "enabled": self.enabled,
        }

    def agent_turn(self, messages: list[Any], tools: list[dict], *, final_only=False, require_action=False, output_tokens=2200):
        """One native model/tool turn. The caller, not a report router, owns continuation."""
        if not self.model:
            raise RuntimeError("真实模型未启用，请检查模型配置。")
        response = self._invoke_with_retry(
            messages, "agent",
            tool_definitions=tools, tool_choice="none" if final_only else "required" if require_action else None,
            output_tokens=output_tokens,
        )
        self.success_count += 1
        self.last_status = "success"
        return response

    @property
    def cancelled(self) -> bool:
        return bool(self.worker and self.worker.cancelled.is_set())

    def cancel(self):
        if self.worker:
            self.worker.cancel()

    def reset_cancellation(self):
        if self.worker:
            self.worker.cancelled.clear()

    def close(self):
        if self.worker:
            self.worker.close()

    def structured_extract(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any] | None = None,
        *,
        stage: str = "structured",
        model_name: str | None = None,
        output_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        """Ask the configured model for a JSON object with a deterministic fallback."""
        self.last_error = None
        if not self.model:
            if self.settings.strict_real:
                self.last_status = "error_no_model"
                raise RuntimeError("真实模型未启用：请检查 DEEPSEEK_API_KEY 和模型配置。")
            self.fallback_count += 1
            self.last_status = "fallback_no_model"
            return fallback
        try:
            response = self._invoke_with_retry(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)], stage,
                model_name=model_name, output_tokens=output_tokens,
            )
            try:
                data = self._parse_json(self._content_text(response))
            except (ValueError, json.JSONDecodeError):
                if (getattr(response, "response_metadata", None) or {}).get("finish_reason") != "length":
                    raise
                self.attempt_trace[-1]["parse_error"] = "output_truncated"
                cap = self.attempt_trace[-1].get("output_limit", 2000)
                response = self._invoke_with_retry(
                    [SystemMessage(content=system_prompt + "\n上次输出超长被截断。只返回紧凑且完整的JSON，合并重复条目，省略非必要内容。"),
                     HumanMessage(content=user_prompt)], stage, model_name=model_name,
                    output_tokens=max(3000, cap * 2),
                )
                data = self._parse_json(self._content_text(response))
            if not isinstance(data, dict):
                raise ValueError("模型没有返回 JSON 对象")
            self.success_count += 1
            self.last_status = "success"
            return data
        except Exception as exc:  # noqa: BLE001 - model boundary is converted or raised in strict mode
            if self.settings.strict_real:
                self.last_status = "error_provider"
                self.last_error = str(exc)
                raise RuntimeError(f"真实结构化模型调用失败：{exc}") from exc
            self.fallback_count += 1
            self.last_status = "fallback_provider_error"
            self.last_error = str(exc)
            return fallback

    def _invoke_with_retry(
        self, messages: list[Any], stage: str, *,
        tool_definitions: list[dict] | None = None, output_tokens: int | None = None,
        tool_choice: str | None = None,
        model_name: str | None = None,
    ) -> Any:
        """Invoke the provider with one bounded application-level retry."""
        attempts = 2 if self.settings.strict_real else 1
        last_error: Exception | None = None
        prompt_chars = sum(len(str(message.content)) for message in messages)
        for attempt in range(1, attempts + 1):
            self.call_count += 1
            started = time.perf_counter()
            started_at = datetime.now(UTC).isoformat()
            record: dict[str, Any] = {
                "stage": stage,
                "model": model_name or self.settings.model_name,
                "attempt": attempt,
                "started_at": started_at,
                "prompt_chars": prompt_chars,
            }
            if self.capture_io:
                record["input_messages"] = [
                    {"role": message.type, "content": message.content,
                     **({"tool_calls": message.tool_calls} if getattr(message, "tool_calls", None) else {}),
                     **({"tool_call_id": message.tool_call_id} if getattr(message, "tool_call_id", None) else {})}
                    for message in messages
                ]
                if tool_definitions is not None:
                    record["input_tools"] = tool_definitions
            try:
                # Report extraction contains several days and grounded sources;
                # the short planner allowance would truncate its JSON.
                stage_limits = {
                    "agent_delivery_check": 2200,
                    "report_region": 1800,
                    "report_airports": 160,
                    "report_keywords": 256,
                    "report_note_selection": 400,
                    "report_supplement_notes": 400,
                    "report_revision_scope": 1100,
                    "report_experience_selection": 1400,
                    "report_route_feasibility": 1400,
                    "report_requested_places": 1400,
                    "report_mobility_modes": 1000,
                    "report_mobility_evidence": 2200,
                    "report_match_places": 1000,
                    "report_schedule": 3000,
                    "report_extract_places": 2600,
                    "report_food_extract": 2600,
                    "report_cost_evidence": 2200,
                }
                limit = output_tokens or (2000 if model_name or stage.startswith("context_compaction") else
                    stage_limits.get(stage, 3500) if stage.startswith("report_") else
                    2200 if stage == "agent_delivery_check" else self.max_output_tokens)
                model = self.model
                if model_name:
                    model = model.bind(model=model_name)
                if tool_definitions:
                    model = model.bind_tools(tool_definitions, **({"tool_choice": tool_choice} if tool_choice else {}))
                # ChatOpenAI rewrites max_tokens to max_completion_tokens, which
                # DeepSeek currently ignores. extra_body reaches the provider verbatim.
                model = model.bind(extra_body={"thinking": {"type": self.thinking_mode}, "max_tokens": limit})
                record["output_limit"] = limit
                if self.worker is None:
                    from travel_agent.model_worker import ModelWorker
                    self.worker = ModelWorker()
                response = self.worker.invoke(model, messages)
                record.update(
                    duration_ms=int((time.perf_counter() - started) * 1000), status="success"
                )
                record["usage"] = getattr(response, "usage_metadata", None) or {}
                record["provider_model"] = (getattr(response, "response_metadata", None) or {}).get("model_name")
                record["finish_reason"] = (getattr(response, "response_metadata", None) or {}).get("finish_reason")
                record["provider_usage"] = (getattr(response, "response_metadata", None) or {}).get(
                    "token_usage", {}
                )
                record["output_chars"] = len(self._content_text(response))
                if getattr(response, "tool_calls", None):
                    record["tool_calls"] = [
                        {"name": call["name"], "id": call.get("id")} for call in response.tool_calls
                    ]
                if self.capture_io:
                    record["output"] = response.content
                    record["output_tool_calls"] = getattr(response, "tool_calls", [])
                self.attempt_trace.append(record)
                if self.usage_recorder:
                    self.usage_recorder(
                        {k: v for k, v in record.items() if k not in {"input_messages", "input_tools", "output", "output_tool_calls"}}
                    )
                return response
            except Exception as exc:  # noqa: BLE001 - provider boundary
                last_error = exc
                record.update(
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    status=type(exc).__name__,
                )
                self.attempt_trace.append(record)
                if self.usage_recorder:
                    self.usage_recorder(
                        {k: v for k, v in record.items() if k not in {"input_messages", "input_tools", "output", "output_tool_calls"}}
                    )
                if getattr(exc, "status_code", None) in {400, 401, 402, 403}:
                    break
                if attempt < attempts:
                    continue
        assert last_error is not None
        raise last_error

    @staticmethod
    def _content_text(response: Any) -> str:
        content = response.content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").removeprefix("json").removesuffix("```").strip()
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(cleaned[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("模型返回值不是 JSON object")
        return value
