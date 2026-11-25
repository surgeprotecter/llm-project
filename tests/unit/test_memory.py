from datetime import UTC, datetime, timedelta

from travel_agent.context import ContextBuilder
from travel_agent.memory import MemoryRepository
from travel_agent.memory.episode import build_episode, episode_line
from travel_agent.memory.extract import extract_memories
from travel_agent.memory.repository import _bm25, tokens
from travel_agent.models import MemoryItem, TripState


def test_extract_preferences_with_slots() -> None:
    items = extract_memories("我吃素，不吃海鲜，节奏轻松一点，预算控制在8000元，还喜欢博物馆。", user_id="u1", trip_id="t1")
    keys = {item.key for item in items}
    assert {"diet.vegetarian", "diet.exclude.海鲜", "pace", "budget.level", "interest.博物馆"} <= keys
    pace = next(item for item in items if item.key == "pace")
    assert pace.structured_value["value"] == "节奏轻松"
    budget = next(item for item in items if item.key == "budget.level")
    assert budget.structured_value["value"] == "预算约8000元"
    assert all(item.source == "user" and item.confidence >= 0.7 for item in items)


def test_extract_skips_third_party_and_second_hand() -> None:
    assert extract_memories("我朋友说他喜欢吃辣，上次说不吃辣。") == []
    assert extract_memories("随便看看") == []


def test_extract_trip_scope_only_when_user_says_so() -> None:
    scoped = extract_memories("这次不要太早出发。", user_id="u1", trip_id="t1")
    assert scoped and scoped[0].trip_id == "t1"
    global_item = extract_memories("我每次都不想早起。", user_id="u1", trip_id="t1")
    assert global_item and global_item[0].trip_id is None


def test_remember_refreshes_same_slot_and_supersedes_changes(database) -> None:
    repository = MemoryRepository(database)
    first = repository.remember(MemoryItem(
        user_id="u1", kind="preference", content="不吃辣", key="diet.spicy",
        structured_value={"key": "diet.spicy", "value": "不吃辣"},
    ))
    refreshed = repository.remember(MemoryItem(
        user_id="u1", kind="preference", content="不吃辣（再次确认）", key="diet.spicy",
        structured_value={"key": "diet.spicy", "value": "不吃辣"},
    ))
    assert refreshed.memory_id == first.memory_id
    assert refreshed.confidence > 0.8
    assert len(database.list_memories("u1")) == 1

    changed = repository.remember(MemoryItem(
        user_id="u1", kind="preference", content="现在能吃一点辣", key="diet.spicy",
        structured_value={"key": "diet.spicy", "value": "能吃一点辣"},
    ))
    active = database.list_memories("u1")
    assert changed.memory_id != first.memory_id
    assert changed.supersedes == first.memory_id
    assert [item.memory_id for item in active] == [changed.memory_id]


def test_retrieval_uses_bm25_recency_and_mmr(database) -> None:
    now = datetime.now(UTC)
    database.save_memory(MemoryItem(user_id="u2", kind="preference", content="不吃辣", key="diet.spicy", importance=0.9))
    database.save_memory(MemoryItem(user_id="u2", kind="preference", content="喜欢自然风光", importance=0.6,
                                   updated_at=now - timedelta(days=200)))
    database.save_memory(MemoryItem(user_id="u2", kind="preference", content="预算约5000元", importance=0.7))
    database.save_memory(MemoryItem(user_id="u2", kind="episode", content="成都 4天", importance=0.9))
    repository = MemoryRepository(database)
    picked = repository.candidates("u2", None, "成都美食预算", "known_destination")
    assert picked
    assert all(item.kind != "episode" for item in picked)
    assert picked[0].content in {"不吃辣", "预算约5000元"}
    assert len(picked) <= 12

    bm25 = _bm25(tokens("成都美食"), [tokens("成都美食"), tokens("北京交通")])
    assert bm25[0] > bm25[1]


def test_episode_built_and_rendered(database) -> None:
    state = {
        "user_id": "u3",
        "trip_id": "t3",
        "current_query": "帮我做一份成都四天的行程",
        "run_status": "completed",
        "request_scope": {"scenario": "execute", "artifact": "plan"},
        "trip_state": {
            "destinations": ["成都"], "duration_days": 4, "traveler_count": 2,
            "travel_dates": {"start_date": "2026-12-01", "end_date": "2026-12-04"},
            "budget": {"amount": 8000, "currency": "CNY"},
        },
        "semantic_summary": {"decisions": ["住春熙路附近", "第3天去都江堰"], "findings": ["熊猫基地需提前预约"]},
        "task_plan": [{"title": "订酒店", "status": "pending"}],
        "plan_approval": {"status": "completed"},
    }
    episode = build_episode(state)
    assert episode is not None
    assert episode.kind == "episode"
    assert episode.trip_id == "t3"
    assert episode.structured_value["decisions"] == ["住春熙路附近", "第3天去都江堰"]
    assert episode.valid_until is not None and episode.valid_until.year == 2027
    line = episode_line(episode)
    assert "成都" in line and "目标" in line and "未完成" in line
    assert build_episode({"user_id": "u4", "trip_state": {}}) is None


def test_context_has_labelled_memory_and_recent_trips(database, settings) -> None:
    database.save_memory(MemoryItem(user_id="u5", kind="preference", content="不吃辣", key="diet.spicy", importance=0.9))
    database.save_memory(MemoryItem(
        user_id="u5", kind="episode", trip_id="t5", content="成都 4天2人",
        structured_value={"title": "成都", "decisions": ["住春熙路"], "open_items": ["订酒店"]},
        importance=0.8,
    ))
    bundle = ContextBuilder(database, settings).build(
        thread_id="u5:t6",
        query="成都吃什么",
        trip_state=TripState(user_id="u5", trip_id="t6", destinations=["成都"]),
        phase="known_destination",
    )
    assert "<user_memory>" in bundle.text
    assert "不吃辣" in bundle.text and "来源: 用户自述" in bundle.text
    assert "<recent_trips>" in bundle.text and "成都" in bundle.text
    assert bundle.manifest.selected_trip_ids
