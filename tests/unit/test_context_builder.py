from datetime import UTC, datetime, timedelta

from travel_agent.context import ContextBuilder
from travel_agent.models import EvidenceItem, MemoryItem, TripState


def test_context_builder_keeps_relevant_memory_and_excludes_stale(database, settings) -> None:
    database.save_memory(MemoryItem(user_id="u1", trip_id="t1", kind="preference", content="偏好慢旅行和自然风景", importance=0.9))
    fresh = EvidenceItem(source_name="fresh", content="成都有美食和轻松路线", related_entities=["known_destination"])
    stale = EvidenceItem(source_name="stale", content="过期酒店价格", valid_until=datetime.now(UTC) - timedelta(hours=1))
    builder = ContextBuilder(database, settings)
    bundle = builder.build(
        thread_id="u1:t1",
        query="成都慢旅行",
        trip_state=TripState(user_id="u1", trip_id="t1", origin="香港", destinations=["成都"], duration_days=4),
        phase="known_destination",
        recent_messages=["user: 我喜欢自然风景"],
        evidence=[fresh, stale],
        exposed_tools=["search_attractions"],
    )
    assert "偏好慢旅行" in bundle.text
    assert fresh.evidence_id in bundle.manifest.selected_evidence_ids
    assert stale.evidence_id not in bundle.manifest.selected_evidence_ids
    assert bundle.manifest.exposed_tools == ["search_attractions"]


def test_context_builder_redacts_xhs_access_token_but_keeps_note_identity(database, settings) -> None:
    evidence = EvidenceItem(
        source_name="xiaohongshu",
        content="成都旅行攻略，适合慢节奏游玩",
        related_entities=["known_destination", "xiaohongshu__search_feeds"],
        structured_data={
            "feeds": [
                {"id": "note-123", "xsecToken": "private-xhs-token", "title": "成都慢旅行攻略"}
            ]
        },
    )
    bundle = ContextBuilder(database, settings).build(
        thread_id="u1:t1",
        query="成都旅行攻略",
        trip_state=TripState(user_id="u1", trip_id="t1", destinations=["成都"]),
        phase="known_destination",
        evidence=[evidence],
    )

    assert "note-123" in bundle.text
    assert "成都慢旅行攻略" in bundle.text
    assert "private-xhs-token" not in bundle.text
