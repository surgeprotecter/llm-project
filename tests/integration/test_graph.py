from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from travel_agent.graph import build_graph


def invoke(graph, query: str, user_id: str = "u1", trip_id: str = "t1"):
    thread_id = f"{user_id}:{trip_id}"
    return graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "user_id": user_id,
            "trip_id": trip_id,
            "thread_id": thread_id,
            "current_query": query,
        },
        {"configurable": {"thread_id": thread_id}},
    )


def test_destination_discovery_generates_report(database, settings) -> None:
    graph, _, _ = build_graph(settings, database)
    result = invoke(graph, "我从香港出发，想去一个不太热的地方玩五天，两个人预算12000元。")
    assert result["trip_state"]["scenario"] == "destination_discovery"
    assert result["trip_state"]["destinations"]
    assert result["assistant_response"]
    assert database.latest_report("t1") is not None
    manifest = database.latest_context_manifest("u1:t1")
    assert manifest and manifest["selected_evidence_ids"]


def test_known_destination_generates_itinerary(database, settings) -> None:
    graph, _, _ = build_graph(settings, database)
    result = invoke(graph, "我从香港出发去成都玩五天，两个人预算12000元，喜欢美食。")
    assert result["trip_state"]["scenario"] == "known_destination"
    assert result["trip_state"]["selected_places"]
    assert result["trip_state"]["hotel_options"]
    assert result["plan"]["itinerary"]
    assert len(result["plan"]["itinerary"]) == 5
    assert result["plan"]["weather"]["temperature"]
    executions = database.list_tool_executions("u1:t1")
    assert any(item["tool_name"] == "plan_route" for item in executions)
    scheduled_places = [
        place.get("name")
        for day in result["plan"]["itinerary"]
        for place in day["activities"]
        if place.get("name")
    ]
    assert len(scheduled_places) == len(set(scheduled_places))
    manifest = database.latest_context_manifest("u1:t1")
    assert manifest and len(manifest["selected_evidence_ids"]) >= 5


def test_graph_can_resume_after_clarification(database, settings) -> None:
    graph, _, _ = build_graph(settings, database)
    thread_id = "u1:clarification"
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="我从香港出发去成都玩五天，预算12000元。")],
            "user_id": "u1",
            "trip_id": "clarification",
            "thread_id": thread_id,
            "current_query": "我从香港出发去成都玩五天，预算12000元。",
        },
        config,
    )
    assert result.get("__interrupt__")
    resumed = graph.invoke(Command(resume="两个人"), config)
    assert not resumed.get("__interrupt__")
    assert resumed["trip_state"]["traveler_count"] == 2
    assert resumed["assistant_response"]


def test_clarification_reuses_duration_and_range_budget(database, settings) -> None:
    graph, _, _ = build_graph(settings, database)
    thread_id = "u1:danang"
    config = {"configurable": {"thread_id": thread_id}}
    first = graph.invoke(
        {
            "messages": [HumanMessage(content="我要去岘港，两人，从香港出发，4日游")],
            "user_id": "u1",
            "trip_id": "danang",
            "thread_id": thread_id,
            "current_query": "我要去岘港，两人，从香港出发，4日游",
        },
        config,
    )
    assert first.get("__interrupt__")
    assert "旅行天数" not in first["__interrupt__"][0].value["question"]
    resumed = graph.invoke(Command(resume="4日游，2000-3000元"), config)
    assert not resumed.get("__interrupt__")
    assert resumed["trip_state"]["duration_days"] == 4
    assert resumed["trip_state"]["budget"]["amount"] == 2500


def test_non_travel_request_stops_safely(database, settings) -> None:
    graph, _, _ = build_graph(settings, database)
    result = invoke(graph, "请帮我写一个排序算法")
    assert result["trip_state"]["scenario"] == "general_planner"
    assert "安全停止" in result["assistant_response"]


def test_greeting_gets_conversational_response(database, settings) -> None:
    graph, _, _ = build_graph(settings, database)
    result = invoke(graph, "你好", user_id="u1", trip_id="greeting")
    assert "你好" in result["assistant_response"]
    assert not database.list_tool_executions("u1:greeting")


def test_graph_uses_configured_mcp_search(database, settings, tmp_path) -> None:
    server_path = Path(__file__).parents[1] / "fixtures" / "mcp_demo_server.py"
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "demo",
                        "command": sys.executable,
                        "args": [str(server_path)],
                        "allowed_tools": ["brave_web_search"],
                        "aliases": {"brave_web_search": "web_search"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    graph, registry, _ = build_graph(replace(settings, mcp_config_path=config_path), database)

    result = invoke(graph, "我从香港出发去成都玩五天，两个人预算12000元。", trip_id="mcp-trip")
    evidence = database.get_evidence(result["trip_state"]["evidence_ids"])

    assert registry.has("web_search")
    assert any(item.source_type == "mcp" and item.source_name == "demo" for item in evidence)


def test_real_mode_does_not_register_mock_tools(database, settings, tmp_path) -> None:
    server_path = Path(__file__).parents[1] / "fixtures" / "mcp_demo_server.py"
    config_path = tmp_path / "mcp-real.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "demo",
                        "command": sys.executable,
                        "args": [str(server_path)],
                        "allowed_tools": ["brave_web_search"],
                        "aliases": {"brave_web_search": "web_search"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    _, registry, _ = build_graph(replace(settings, mcp_config_path=config_path, tool_mode="real"), database)

    assert list(registry.tools) == ["web_search"]


def test_planner_question_routes_to_user_interrupt(monkeypatch, database, settings) -> None:
    class AskUserGateway:
        enabled = False

        def __init__(self, _settings):
            pass

        def structured_extract(self, _system, _prompt, fallback=None, *, stage=None):
            if stage == "plan":
                return {
                    "action": "ask_user",
                    "tool_name": None,
                    "arguments": {},
                    "reason": "测试参数不完整",
                    "question": "请补充出行日期。",
                }
            return fallback

    monkeypatch.setattr("travel_agent.graph.builder.ModelGateway", AskUserGateway)
    graph, _, _ = build_graph(settings, database)

    result = invoke(graph, "我从香港出发去成都玩三天，两个人预算5000元。", trip_id="ask-user-route")

    assert result.get("__interrupt__")
    assert result["__interrupt__"][0].value["question"] == "请补充出行日期。"
