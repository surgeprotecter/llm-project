from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: Literal["mock", "api", "cli", "skill", "mcp", "web", "user"] = "mock"
    source_name: str
    source_url: str | None = None
    query: str | None = None
    content: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    confidence: float = 0.7
    related_entities: list[str] = Field(default_factory=list)
    is_stale: bool = False
