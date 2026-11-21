from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    trip_id: str
    thread_id: str
    current_query: str
    phase: str
    trip_state: dict[str, Any]
    context_text: str
    context_manifest: dict[str, Any]
    planner_tool_results_context: str
    tool_results: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    plan: dict[str, Any]
    assistant_response: str
    pending_question: str
    step_count: int
    error: str
    requirement_decision: dict[str, Any]
    capability_catalog: list[dict[str, Any]]
    selected_capability: str
    tool_decision: dict[str, Any]
    planner_round: int
    loaded_tool_spec: dict[str, Any]
    resume_pending: bool
    agent_messages: list[AnyMessage]
    loaded_tools: list[str]
    loaded_skills: dict[str, str]
    pending_calls: list[dict[str, Any]]
    report_workspace: dict[str, Any]
    turn_results: list[dict[str, Any]]
    agent_done: bool
    agent_error: str
    final_checks: int
    action_cache: dict[str, Any]
    protocol_repairs: int
    initial_report_version: int
    checked_artifact_version: int
    finalization_rounds: int
    local_finalization: bool
    task_plan: list[dict[str, Any]]
    active_goal: str
    root_goal: str
    request_scope: dict[str, Any]
    plan_approval: dict[str, Any]
    pending_plan: dict[str, Any]
    unpresented_plan: dict[str, Any]
    export_source: dict[str, Any]
    previous_run_status: str
    stalled_batches: int
    delivery_feedback: dict[str, Any]
    run_status: str
    semantic_summary: dict[str, Any]
    summary_version: int
    repeat_reads: dict[str, Any]
    read_progress: dict[str, Any]
    selected_results: list[dict[str, Any]]
    action_observations: dict[str, str]
    needs_action: bool
    needs_progress: bool
    research_checkpoint: str
