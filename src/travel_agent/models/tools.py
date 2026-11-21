from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    success: bool
    tool_name: str
    result_id: str | None = None
    reused: bool = False
    data: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    source_type: str = "mock"
    source_name: str = ""
    source_url: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    error_type: str | None = None
    error_message: str | None = None


class ToolExecution(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    thread_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: ToolResult
    duration_ms: int = 0
    attempt: int = 1
