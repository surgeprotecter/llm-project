from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = "default"
    trip_id: str | None = None
    kind: Literal["preference", "decision", "episode", "procedure", "fact"] = "fact"
    content: str
    structured_value: dict[str, Any] = Field(default_factory=dict)
    source: Literal["user", "tool", "inferred", "system"] = "user"
    confidence: float = 1.0
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    phase: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    supersedes: str | None = None
    is_active: bool = True
