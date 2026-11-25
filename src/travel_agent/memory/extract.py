"""Deterministic long-term memory extraction.

The planner already has a ``remember`` tool for memories the user asks to keep
explicitly.  This module complements it by recognising stable, first-person
preferences in the user's own wording (diet, pace, budget level, ...), so the
system does not depend on the model remembering to call a tool every turn.

Only statements about the user are extracted; quotes, second-hand remarks and
temporary trip details are ignored on purpose.  Extraction is intentionally
conservative: a missed preference is cheaper than a wrong one.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from travel_agent.models import MemoryItem

# (key, pattern, canonical value, importance)
_RULES: list[tuple[str, str, str, float]] = [
    ("diet.spicy", r"不能吃辣|不吃辣|吃不了辣|怕辣", "不吃辣", 0.85),
    ("diet.spicy", r"无辣不欢|爱吃辣|能吃辣|喜欢辣", "喜欢吃辣", 0.7),
    ("diet.vegetarian", r"吃素|素食|不吃肉", "素食", 0.85),
    ("diet.exclude", r"不吃(海鲜|牛肉|猪肉|羊肉|香菜|内脏|生食)", "", 0.85),
    ("avoid.early", r"不要?早班|别太早|不要太?早|别那么早|不(想|要|能)早起|睡到自然醒|别安排早上", "避免早班/早起", 0.75),
    ("pace", r"节奏(?:要|安排)?(?:轻松|慢一点|慢点)|不要?排太满|别太赶|轻松一点|慢旅行", "节奏轻松", 0.75),
    ("pace", r"节奏(?:可以|要|想)?(?:紧凑|紧一点|快一点)|紧凑一点|行程排满|尽量多逛|想多去几个地方", "节奏紧凑", 0.65),
    ("hotel.style", r"酒店(?:要|喜欢|偏好|希望)(安静|市中心|设计感|亲子|民宿|干净|交通方便)", "", 0.7),
    ("budget.level", r"预算(?:控制在|不超过|最多|大概|约)?\s*(\d{3,6})\s*元", "", 0.7),
    ("schedule.start", r"希望(?:每天)?\s*(\d{1,2})\s*点(?:出发|开始)", "", 0.65),
    ("accessibility", r"腿脚不便|行动不便|不能久走|走不了太多路", "少走路", 0.9),
    ("profile.name", r"(?:记住)?我叫\s*([\u4e00-\u9fa5A-Za-z]{1,12}?)(?:[，,。！!\s]|$)", "", 0.8),
]

_INTEREST = re.compile(r"(?:喜欢|偏爱|爱看|爱逛|想去)(建筑|博物馆|美术馆|美食|自然|山野|海|购物|历史|艺术|摄影|咖啡|夜景|温泉)")
_INTEREST_NEG = re.compile(r"(?:不喜欢|讨厌|没兴趣)(建筑|博物馆|美术馆|美食|自然|购物|历史|艺术|摄影|咖啡|夜景|温泉)")

_SKIP_CONTEXT = re.compile(r"上次说|听说|别人|朋友说|他说|她说|网上说|据说|同事说|家人说")
_TRIP_SCOPE = re.compile(r"这次|本次|本行程|这趟|此次|这周|这几天")

# Chinese statements are split on punctuation and conjunction words so that a
# clause about someone else is not attributed to the user.
_CLAUSE_SPLIT = re.compile(r"[。！？!?；;\n]|但是|不过|而且|然后|另外")


def _clauses(text: str) -> list[str]:
    return [part.strip() for part in _CLAUSE_SPLIT.split(text or "") if part.strip()]


def _third_person(clause: str, start: int) -> bool:
    """True when the matched phrase is attributed to someone other than the user."""
    prefix = clause[:start][-8:]
    return bool(re.search(r"[他她]|朋友|同事|家人|孩子|老人|别人", prefix))


def extract_memories(
    text: str,
    *,
    user_id: str = "default",
    trip_id: str | None = None,
    phase: str | None = None,
    limit: int = 6,
) -> list[MemoryItem]:
    """Extract stable preferences from one user message.

    ``trip_id`` is set only when the user scopes the preference to this trip
    (这次/本行程/...); otherwise the memory is global and reused in later trips.
    """
    if not text or not text.strip():
        return []
    scoped = bool(_TRIP_SCOPE.search(text))
    scope = trip_id if scoped else None
    found: dict[str, MemoryItem] = {}

    for key, pattern, value, importance in _RULES:
        for clause in _clauses(text):
            if _SKIP_CONTEXT.search(clause):
                continue
            match = re.search(pattern, clause)
            if not match:
                continue
            if _third_person(clause, match.start()):
                continue
            if key == "diet.exclude":
                key, value = f"diet.exclude.{match.group(1)}", f"不吃{match.group(1)}"
            elif key == "hotel.style":
                value = f"酒店偏好{match.group(1)}"
            elif key == "budget.level":
                amount = int(match.group(1))
                if amount < 500:
                    continue
                value = f"预算约{amount}元"
            elif key == "schedule.start":
                hour = int(match.group(1))
                if not 4 <= hour <= 14:
                    continue
                value = f"{hour}点后出发"
            elif key == "profile.name":
                value = match.group(1).strip()
                if not value or value in {"什么", "谁", "你"}:
                    continue
            if key in found:
                continue
            content = f"{value}（用户自述，{datetime.now(UTC).date().isoformat()}）"
            found[key] = MemoryItem(
                user_id=user_id,
                trip_id=scope,
                kind="preference",
                content=content,
                key=key,
                structured_value={"key": key, "value": value},
                source="user",
                importance=importance,
                confidence=0.8,
                phase=phase,
                tags=["auto-extracted", "trip" if scoped else "global"],
            )
            break

    for clause in _clauses(text):
        if _SKIP_CONTEXT.search(clause):
            continue
        for regex, kind in ((_INTEREST, "interest"), (_INTEREST_NEG, "avoid")):
            match = regex.search(clause)
            if match and not _third_person(clause, match.start()):
                key = f"{kind}.{match.group(1)}"
                if key not in found:
                    value = f"{'喜欢' if kind == 'interest' else '不喜欢'}{match.group(1)}"
                    found[key] = MemoryItem(
                        user_id=user_id,
                        trip_id=scope,
                        kind="preference",
                        content=f"{value}（用户自述）",
                        key=key,
                        structured_value={"key": key, "value": value},
                        source="user",
                        importance=0.65,
                        confidence=0.75,
                        phase=phase,
                        tags=["auto-extracted"],
                    )

    return list(found.values())[:limit]
