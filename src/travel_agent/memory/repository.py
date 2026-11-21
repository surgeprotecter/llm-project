from __future__ import annotations

import re
from datetime import UTC, datetime

from travel_agent.models import MemoryItem
from travel_agent.persistence import Database


class MemoryRepository:
    def __init__(self, database: Database):
        self.database = database

    def save(self, item: MemoryItem) -> None:
        self.database.save_memory(item)

    def candidates(self, user_id: str, trip_id: str | None, query: str, phase: str | None = None) -> list[MemoryItem]:
        now = datetime.now(UTC)
        terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
        items = self.database.list_memories(user_id, trip_id)
        scored: list[tuple[float, MemoryItem]] = []
        for item in items:
            if item.valid_until and item.valid_until < now:
                continue
            if phase and item.phase and item.phase != phase:
                phase_score = 0.0
            else:
                phase_score = 1.0
            item_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", item.content.lower()))
            overlap = len(terms & item_terms) / max(len(terms), 1)
            score = 0.45 * overlap + 0.2 * phase_score + 0.2 * item.importance + 0.15 * item.confidence
            scored.append((score, item))
        scored.sort(key=lambda value: value[0], reverse=True)
        return [item for score, item in scored if score > 0.05][:12]
