from __future__ import annotations

import math
import re
from datetime import UTC, datetime

from travel_agent.models import MemoryItem
from travel_agent.persistence import Database

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Chinese-aware tokens: CJK bigrams (plus single chars for short runs) + latin words."""
    value = (text or "").lower()
    result: list[str] = []
    for run in _CJK_RUN.findall(value):
        if len(run) == 1:
            result.append(run)
        else:
            result.extend(run[index:index + 2] for index in range(len(run) - 1))
    result.extend(_LATIN.findall(value))
    return result


def _bm25(query: list[str], corpus: list[list[str]], k1: float = 1.2, b: float = 0.75) -> list[float]:
    if not corpus:
        return []
    avgdl = sum(len(doc) for doc in corpus) / len(corpus) or 1.0
    df: dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    total = len(corpus)
    scores: list[float] = []
    for doc in corpus:
        tf: dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in query:
            if term not in tf:
                continue
            idf = math.log(1 + (total - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(doc) / avgdl)
            score += idf * tf[term] * (k1 + 1) / denom
        scores.append(score)
    return scores


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class MemoryRepository:
    def __init__(self, database: Database):
        self.database = database

    def save(self, item: MemoryItem) -> None:
        self.database.save_memory(item)

    # ------------------------------------------------------------------ remember
    @staticmethod
    def _value(item: MemoryItem) -> str:
        return str(item.structured_value.get("value", "")).strip()

    def remember(self, item: MemoryItem) -> MemoryItem:
        """Persist with slot semantics: same slot+value refreshes, new value supersedes."""
        key = item.key or str(item.structured_value.get("key") or "")
        item.key = key
        if key:
            same_slot = [
                old for old in self.database.list_memories(item.user_id, item.trip_id)
                if (old.key or str(old.structured_value.get("key") or "")) == key
                and (old.trip_id or None) == (item.trip_id or None)
            ]
            for old in same_slot:
                if self._value(old) == self._value(item):
                    old.content = item.content
                    old.confidence = min(1.0, max(old.confidence, item.confidence) + 0.05)
                    old.importance = max(old.importance, item.importance)
                    old.structured_value = item.structured_value
                    old.updated_at = datetime.now(UTC)
                    self.database.save_memory(old)
                    return old
            if same_slot:
                item.supersedes = same_slot[0].memory_id
                for old in same_slot:
                    old.is_active = False
                    self.database.save_memory(old)
        self.database.save_memory(item)
        return item

    def forget(self, memory_id: str, user_id: str) -> bool:
        return self.database.delete_memory(memory_id, user_id)

    def all(self, user_id: str) -> list[MemoryItem]:
        return self.database.list_all_memories(user_id, include_inactive=False)

    def episodes(self, user_id: str, limit: int = 3) -> list[MemoryItem]:
        items = [
            item for item in self.database.list_all_memories(user_id, include_inactive=False)
            if item.kind == "episode"
        ]
        items.sort(key=lambda item: item.created_at, reverse=True)
        return items[:limit]

    # ----------------------------------------------------------------- retrieval
    def candidates(
        self,
        user_id: str,
        trip_id: str | None,
        query: str,
        phase: str | None = None,
        *,
        limit: int = 12,
    ) -> list[MemoryItem]:
        now = datetime.now(UTC)
        query_tokens = tokens(query)
        items: list[MemoryItem] = []
        for item in self.database.list_memories(user_id, trip_id):
            if item.valid_until and item.valid_until < now:
                continue
            if item.kind == "episode":
                continue
            items.append(item)
        if not items:
            return []
        corpus = [tokens(item.content) for item in items]
        bm25 = _bm25(query_tokens, corpus) if query_tokens else [0.0] * len(items)
        peak = max(bm25) or 1.0
        scored: list[tuple[float, set[str], MemoryItem]] = []
        for raw, item in zip(bm25, items, strict=True):
            relevance = raw / peak
            age_days = max(0.0, (now - item.updated_at).total_seconds() / 86400)
            recency = math.exp(-age_days / 30)
            phase_score = 1.0 if not (phase and item.phase and item.phase != phase) else 0.0
            scope = 1.0 if item.trip_id in (None, trip_id) else 0.6
            score = (
                0.40 * relevance
                + 0.15 * recency
                + 0.15 * item.importance
                + 0.15 * item.confidence
                + 0.15 * phase_score * scope
            )
            if score > 0.05:
                scored.append((score, set(tokens(item.content)), item))
        scored.sort(key=lambda value: value[0], reverse=True)

        # Maximal marginal relevance: keep preferences that add new information.
        selected: list[MemoryItem] = []
        pool = scored[:24]
        while pool and len(selected) < limit:
            best_index, best_score = 0, -1.0
            for index, (score, item_tokens, _item) in enumerate(pool):
                overlap = max((_jaccard(item_tokens, chosen) for chosen in (
                    set(tokens(picked.content)) for picked in selected
                )), default=0.0)
                mmr = 0.7 * score - 0.3 * overlap
                if mmr > best_score:
                    best_index, best_score = index, mmr
            _score, _tokens, chosen_item = pool.pop(best_index)
            selected.append(chosen_item)
        return selected
