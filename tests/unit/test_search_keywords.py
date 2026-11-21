from travel_agent.graph.agent_nodes import (
    _ground_itinerary,
    _research_notes_from_results,
    _tool_results_for_context,
    _xhs_search_keywords,
    _xhs_token_for_feed,
)


def test_xhs_keywords_use_short_destination_query_and_drop_long_model_output() -> None:
    keywords = _xhs_search_keywords(
        "成都",
        "成都旅游攻略 本地人推荐 避坑",
        ["成都小众景点", "成都本地人周末避坑路线建议", "成都小众景点"],
    )

    assert keywords == ["成都旅游攻略", "成都小众景点"]
    assert all(len(keyword) <= 12 for keyword in keywords)


def test_planner_tool_context_is_bounded_and_keeps_result_samples() -> None:
    context = _tool_results_for_context(
        [
            {
                "tool_name": "xiaohongshu__search_feeds",
                "success": True,
                "source_name": "xiaohongshu",
                "summary": "20 notes found",
                "data": {
                    "feeds": [
                        {
                            "title": "成都攻略",
                            "desc": "景点建议" + "内容" * 5000,
                            "feed_id": "note-1",
                            "xsecToken": "opaque-token-value",
                        },
                        {"title": "本地美食", "feed_id": "note-2"},
                    ]
                },
            }
        ]
    )

    assert len(context) <= 8000
    assert "成都攻略" in context
    assert "note-1" in context
    assert "opaque-token-value" not in context
    assert "full results remain in evidence store" not in context


def test_xhs_detail_token_is_resolved_from_matching_feed_not_model_or_user() -> None:
    results = [
        {
            "tool_name": "xiaohongshu__search_feeds",
            "data": {
                "feeds": [
                    {"id": "note-1", "xsecToken": "opaque-token-1"},
                    {"id": "note-2", "xsecToken": "opaque-token-2"},
                ]
            },
        }
    ]

    assert _xhs_token_for_feed(results, "note-2") == "opaque-token-2"
    assert _xhs_token_for_feed(results, "missing-note") is None


def test_xhs_report_merges_detail_text_and_keeps_only_verbatim_places() -> None:
    results = [
        {
            "tool_name": "xiaohongshu__search_feeds",
            "data": {
                "feeds": [
                    {
                        "id": "note-1",
                        "xsecToken": "private-token",
                        "noteCard": {"displayTitle": "成都慢旅行", "user": {"nickname": "作者"}},
                    }
                ]
            },
        },
        {
            "tool_name": "xiaohongshu__get_feed_detail",
            "data": {
                "feed_id": "note-1",
                "data": {
                    "note": {
                        "noteId": "note-1",
                        "xsecToken": "private-token",
                        "title": "成都慢旅行",
                        "desc": "可以慢慢逛宽窄巷子，傍晚去锦里散步。",
                    }
                },
            },
        },
    ]

    notes = _research_notes_from_results(results)
    itinerary = _ground_itinerary(
        {
            "days": [
                {
                    "title": "慢节奏老街",
                    "activities": [
                        {"name": "宽窄巷子", "reason": "笔记建议慢慢逛。", "source_note": "成都慢旅行"},
                        {"name": "虚构景点", "reason": "不应保留。"},
                    ],
                }
            ]
        },
        notes,
        4,
    )

    assert len(notes) == 1
    assert "宽窄巷子" in notes[0]["content"]
    assert "private-token" not in str(notes)
    assert [activity["name"] for activity in itinerary[0].activities] == ["宽窄巷子"]
    assert "待地图核验" in itinerary[0].title
