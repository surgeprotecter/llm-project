from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from travel_agent.config import Settings
from travel_agent.context import ContextBuilder
from travel_agent.graph.agent_nodes import make_agent_nodes
from travel_agent.graph.state import GraphState
from travel_agent.llm import ModelGateway
from travel_agent.mcp import MCPServerManager, MCPToolAdapter, load_mcp_configs
from travel_agent.persistence import Database
from travel_agent.reports import ReportRenderer
from travel_agent.tools import ToolRegistry
from travel_agent.tools.hotel_quote import HotelQuoteTool
from travel_agent.tools.live_http import build_live_http_tools
from travel_agent.tools.mock_travel import build_mock_tools
from travel_agent.tools.web_search import FreeWebSearchTool


def build_graph(settings: Settings, database: Database):
    context_builder = ContextBuilder(database, settings)
    gateway = ModelGateway(settings)
    registry = ToolRegistry(database, settings.max_tool_retries)
    if settings.tool_mode == "mock":
        for tool in build_mock_tools():
            registry.register(tool)
    elif settings.tool_mode != "real":
        raise ValueError("TRAVEL_AGENT_TOOL_MODE must be 'mock' or 'real'")
    if settings.tool_mode == "real":
        for tool in build_live_http_tools():
            registry.register(tool)
        registry.register(HotelQuoteTool())
    mcp_manager = MCPServerManager(load_mcp_configs(settings.mcp_config_path))
    registry.mcp_manager = mcp_manager
    for metadata in mcp_manager.discover_tools():
        registry.register(MCPToolAdapter(mcp_manager, metadata))
    registry.integration_errors.update(mcp_manager.discovery_errors)
    if settings.tool_mode == "real":
        registry.register(FreeWebSearchTool(registry.tools.pop("web_search", None)))
    renderer = ReportRenderer(database, settings.output_dir)
    if settings.strict_real:
        from travel_agent.graph.runtime import make_runtime_nodes
        nodes = make_runtime_nodes(database, registry, gateway, renderer, settings)
        builder = StateGraph(GraphState)
        for name, node in nodes.items():
            builder.add_node(name, node)
        builder.add_edge(START, "prepare_turn")
        builder.add_edge("prepare_turn", "model_turn")
        builder.add_conditional_edges(
            "model_turn",
            lambda state: "done" if state.get("agent_done") else "tools" if state.get("pending_calls") else "continue",
            {"done": END, "tools": "execute_actions", "continue": "model_turn"},
        )
        builder.add_edge("execute_actions", "model_turn")
        connection = sqlite3.connect(settings.database_path, check_same_thread=False)
        from travel_agent.graph.driver import DurableAgentGraph
        graph = builder.compile(
            checkpointer=SqliteSaver(connection), interrupt_after=["model_turn", "execute_actions"],
        )
        return DurableAgentGraph(graph, database, gateway, registry), registry, gateway
    nodes = make_agent_nodes(database, context_builder, registry, gateway, renderer, settings)

    builder = StateGraph(GraphState)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "check_requirements")

    def route_requirements(state: GraphState) -> str:
        decision = state.get("requirement_decision", {})
        if state.get("pending_question"):
            return "clarify_with_user"
        if decision.get("intent") in {"greeting", "other"}:
            return "respond_general"
        return "build_context"

    builder.add_conditional_edges("check_requirements", route_requirements, {"clarify_with_user": "clarify_with_user", "respond_general": "respond_general", "build_context": "build_context"})
    builder.add_edge("clarify_with_user", "extract_requirements")

    builder.add_edge("build_context", "plan_next_action")

    def route_plan(state: GraphState) -> str:
        action = state.get("tool_decision", {}).get("action")
        if action == "use_tool":
            return "execute_selected_tool"
        if action == "ask_user":
            if state.get("pending_question") or state.get("tool_decision", {}).get("question"):
                return "clarify_with_user"
            return "respond_general"
        if action == "stop":
            return "respond_general"
        return "build_plan"

    builder.add_conditional_edges(
        "plan_next_action",
        route_plan,
        {
            "execute_selected_tool": "execute_selected_tool",
            "clarify_with_user": "clarify_with_user",
            "respond_general": "respond_general",
            "build_plan": "build_plan",
        },
    )
    builder.add_edge("execute_selected_tool", "assess_tool_result")

    def route_assessment(state: GraphState) -> str:
        action = state.get("tool_decision", {}).get("action")
        if action == "build_plan":
            return "build_plan"
        if action == "stop":
            return "respond_general"
        return "build_context"

    builder.add_conditional_edges(
        "assess_tool_result",
        route_assessment,
        {"build_context": "build_context", "build_plan": "build_plan", "respond_general": "respond_general"},
    )
    builder.add_edge("build_plan", "render_report")
    builder.add_edge("render_report", END)
    builder.add_edge("respond_general", END)

    checkpoint_connection = sqlite3.connect(settings.database_path, check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_connection)
    return builder.compile(checkpointer=checkpointer), registry, gateway
