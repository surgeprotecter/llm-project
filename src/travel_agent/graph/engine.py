"""Model-led execution with durable plans, evidence and report progress."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jsonschema import Draft202012Validator
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from travel_agent.context.agent_context import (
    ContextCapacityError,
    build_agent_messages,
    conservative_tokens,
    dump,
    project,
    recent_lookup_quotes,
    selection_view,
    tool_view,
    transfer_index,
    trip_summary,
)
from travel_agent.context.semantic import compact_context
from travel_agent.graph.delivery import (
    check_reply,
    classify_request,
    reply_task_ready,
    report_allowed,
)
from travel_agent.graph.handoff import saved_material
from travel_agent.graph.reading import (
    exhausted_page,
    first_unread,
    next_args,
    page_info,
    page_target,
    preview_result,
    remember_pages,
)
from travel_agent.models import (
    EvidenceItem,
    MemoryItem,
    ToolExecution,
    ToolResult,
    TravelPlan,
    TripState,
)
from travel_agent.persistence.runtime import canonical, fingerprint
from travel_agent.reports.delivery import delivered_report, receipt, standard_report_ready
from travel_agent.reports.route_contract import (
    approved_plan_view,
    extract_route_contract,
    route_issues,
)
from travel_agent.reports.workbench import report_operation

CORE_TOOLS = {"inspect_tools", "load_skill", "read_state", "read_result", "update_plan"}
READ_TOOLS = {"read_state", "read_result"}
FLIGHT_LOOKUPS = {"google_flights__search_flights", "google_flights__search_dates", "google_flights__search_explore"}
ORIGIN_QUESTION = "你打算从哪个城市或机场出发？如果有几个出发地都方便，也可以一起告诉我。"


def permitted_origins(scope, trip):
    if "departure_origins" in scope:
        return scope["departure_origins"]
    # Previously approved runs did not store a separate departure field.
    return [trip.origin] if trip.origin else []


def resolve_flight_dates(args, request, trip, approved):
    """Repair stale inferred years only when the trip or user determines the date."""
    today = datetime.now(ZoneInfo("Asia/Hong_Kong")).date()
    for key, known in (("departure_date", trip.travel_dates.start_date), ("return_date", trip.travel_dates.end_date)):
        try:
            proposed = date.fromisoformat(args.get(key) or "")
        except (TypeError, ValueError):
            continue
        if proposed >= today:
            continue
        if approved and known and known >= today and (known.month, known.day) == (proposed.month, proposed.day):
            args = {**args, key: known.isoformat()}
        elif not re.search(r"\d{4}|去年|历史|过去|当时", request) and re.search(
            rf"(?<!\d)0?{proposed.month}\s*月\s*0?{proposed.day}(?!\d)", request
        ):
            for year in range(today.year, today.year + 5):
                try:
                    corrected = proposed.replace(year=year)
                except ValueError:
                    continue
                if corrected >= today:
                    args = {**args, key: corrected.isoformat()}
                    break
    return args


def mark_chat_delivery(scope, plan):
    if scope["scenario"] != "execute" or scope["artifact"] != "none" or not plan:
        return
    final = plan[-1]
    if (final["status"] != "completed" and not final.get("completion")
        and re.search(r"(?:整合|汇总|撰写|生成).*(?:行程|报告|攻略|答复)", final["description"])):
        final.update(completion="reply", description="整合已保存结果，在聊天中交付完整答复")


class RuntimeEngine:
    def __init__(self, database, registry, gateway, renderer, settings):
        from travel_agent.graph.runtime import INTERNAL, POLICY, discover_skills
        self.db, self.registry, self.gateway, self.renderer, self.settings = database, registry, gateway, renderer, settings
        self.store, self.policy, self.skills = database.runtime, POLICY, discover_skills()
        self.builtins = {x["function"]["name"]: x for x in INTERNAL}

    def event(self, state, kind, payload):
        self.db.append_event(state["thread_id"], state["user_id"], state["trip_id"], kind, payload)

    def progress(self, kind, payload):
        if self.gateway.progress_callback:
            self.gateway.progress_callback(kind, payload)

    def save(self, state, trip=None):
        if trip:
            state["trip_state"] = trip.model_dump(mode="json")
            self.db.save_trip(trip)
        self.store.save(state["thread_id"], state)

    def tool_spec(self, name):
        """Capability discovery is independent of the current turn's authorization."""
        spec = self.builtins.get(name)
        if spec:
            return spec
        from travel_agent.graph.runtime import external_definition
        return external_definition(spec) if (spec := self.registry.full_spec(name)) else None

    def action_error(self, state, trip, name, args):
        """Authorize the effect, not the discovery or loading of a tool."""
        scope = state["request_scope"]
        scenario = scope["scenario"]
        artifact = scope["artifact"]
        approved = scenario == "execute" and state.get("plan_approval", {}).get("status") == "approved"
        if name in {"inspect_tools", "load_skill", *READ_TOOLS}:
            return None
        is_external = self.registry.has(name) or name == "refresh_tool"
        if name == "export_report":
            allowed = (args.get("operation") == "reexport" and bool(state.get("export_source"))
                       or args.get("operation") == "render" and approved and artifact == "report")
        elif name in {"update_plan", "itinerary_step", "select_result"}:
            allowed = approved
        elif name == "update_trip":
            allowed = scenario not in {"plan_review", "progress"} and artifact != "report_copy"
        elif is_external and scenario == "plan_review":
            allowed = bool(scope.get("preapproval_lookup"))
        else:
            allowed = scenario != "progress" and artifact != "report_copy"
        if not allowed:
            return {"success": False, "error_type": "outside_request_scope",
                    "error": ("本轮只修订提案，还未授权实际查询；把查询写进计划供确认。"
                              if is_external and scenario == "plan_review"
                              else "用户没有授权本轮生成这一文件；按本轮要求在聊天中答复。"
                              if name == "export_report" and approved and artifact == "none"
                              else "当前请求没有授权执行这个动作。"),
                    "next": "按当前用户请求选择可做的动作，或交付答复；不要重试此动作。"}
        target = args.get("name") if name == "refresh_tool" else name
        if name == "refresh_tool" and (not isinstance(target, str) or not self.registry.has(target)):
            return {"success": False, "error_type": "unknown_tool", "error": "刷新目标必须是目录中已接入的外部工具。"}
        origins = permitted_origins(scope, trip)
        external_step = name == "itinerary_step" and args.get("operation") == "transport" and (
            args.get("transport_target") or state.get("report_workspace", {}).get("transport_target") or "all"
        ) != "intercity"
        if (is_external and target in FLIGHT_LOOKUPS or scope.get("requires_departure")
                and (external_step or name == "export_report" and args.get("operation") == "render")) and not origins:
            return {"success": False, "error_type": "missing_user_input", "field": "origin",
                    "error": "缺少用户确认的出发地，未发起交通查询或交付。",
                    "next": "询问用户从哪个城市或机场出发；如果有多个候选，让用户给出可比较的范围。"}
        if external_step and len(origins) > 1 and trip.origin not in origins:
            return {"success": False, "error_type": "unresolved_departure_options",
                    "error": "还未从用户给出的出发地中确定本方案的出发地。", "origin_options": origins,
                    "next": "先按用户的取舍标准比较获准的候选，再用update_trip保存选中的原始名称。"}
        return None

    def export_source(self, state, row):
        references = self.store.index(state["thread_id"], tool_name="deliver_reply", limit=1)
        reply = references[0] if references else None
        if not reply:
            legacy = self.db.connection.execute(
                "SELECT payload_json, created_at FROM tool_executions WHERE thread_id=? AND tool_name=? ORDER BY rowid DESC LIMIT 1",
                (state["thread_id"], "deliver_reply"),
            ).fetchone()
            if legacy:
                reply = {"result_id": json.loads(legacy[0])["result"]["result_id"], "fetched_at": legacy[1]}
        if row and (not reply or row["created_at"] >= reply["fetched_at"]):
            saved = json.loads(row["payload_json"]).get("report") or {}
            if standard_report_ready(saved):
                return {"kind": "report", "version": row["version"]}
            if saved.get("chat_markdown"):
                return {"kind": "chat_file", "version": row["version"]}
        if reply:
            return {"kind": "reply", "result_id": reply["result_id"]}
        return None

    def prepare(self, state):
        from travel_agent.graph.runtime import discover_skills
        self.skills = discover_skills()
        prior = self.store.load(state["thread_id"])
        combined = {**state, **prior}
        self.gateway.turn_usage_start = len(self.gateway.attempt_trace)
        self.gateway.usage_recorder = lambda record: self.event(state, "model.usage", record)
        row = self.db.latest_report(state["trip_id"])
        history = state.get("messages", [])
        exportable = self.export_source(state, row)
        scope, approval_action = classify_request(self.gateway, state["current_query"], history, prior, exportable)
        approval = dict(prior.get("plan_approval", {}))
        export_source = scope.pop("export_source", {})
        pending = prior.get("pending_plan", {})
        revision = prior.get("unpresented_plan") or {}
        if approval.get("status") == "pending":
            pending, approval = approval, {}
        running = approval_action in {"approve", "resume"}
        base_version = row["version"] if row else 0
        if approval_action == "approve":
            if approval:
                self.store.archive(state["thread_id"], {k: prior[k] for k in (
                    "plan_approval", "task_plan", "request_scope", "run_status",
                ) if k in prior})
            approval = {**pending, "status": "approved", "user_confirmation": state["current_query"],
                        "base_report_version": base_version}
            pending = {}
            combined.update(task_plan=[], root_goal=approval["request"], active_goal=scope["deliverable"],
                            semantic_summary={}, agent_messages=[], loaded_tools=[], summary_version=0,
                            repeat_reads={}, read_progress={}, action_observations={})
            self.event(state, "agent.plan_approved", {"version": approval["version"]})
        elif approval_action == "resume":
            approval["scope"] = scope
        if scope["scenario"] == "plan_review":
            requests = revision.get("requests", [revision["request"]] if revision.get("request") else [])
            revision = {"request": state["current_query"], "scope": scope,
                        "requests": list(dict.fromkeys([*requests, state["current_query"]]))[-8:]}
        stored = self.db.get_trip(state["trip_id"])
        trip = TripState.model_validate(combined.get("trip_state") or (stored.model_dump() if stored else {}))
        # Older select_result versions copied the model's free-form purpose into
        # confirmed user facts. Keep explicit update_trip confirmations, not those copies.
        auto, explicit = set(), set()
        for row in self.db.connection.execute("SELECT payload_json FROM tool_executions WHERE thread_id=? AND tool_name IN ('select_result','update_trip')", (state["thread_id"],)):
            record = json.loads(row[0])
            if record["tool_name"] == "select_result":
                auto.add(record["arguments"].get("purpose"))
            else:
                explicit.update(record["arguments"].get("requirements", {}).get("confirmed_decisions", []))
        trip.confirmed_decisions = [d for d in trip.confirmed_decisions if d not in auto or d in explicit]
        trip.user_id, trip.trip_id, trip.thread_id = state["user_id"], state["trip_id"], state["thread_id"]
        trip.current_phase = "general_planner"
        if running and "departure_origins" in scope:
            origins = scope["departure_origins"]
            if len(origins) == 1:
                trip.origin = origins[0]
            elif trip.origin not in origins or approval_action == "approve":
                trip.origin = None
        if running:
            messages = list(combined.get("agent_messages", []))
            if approval_action == "resume" and scope["artifact"] == "report" and not combined.get("report_workspace"):
                # A legacy session may have searched before creating a workspace.
                # Keep the actual choices visible and reuse evidence by ID.
                messages = [m for m in history[:-1] if isinstance(m, HumanMessage)
                            or isinstance(m, AIMessage) and not m.tool_calls
                            and "<html" not in str(m.content).lower()][-6:]
        else:
            # Prior rejected candidates and program feedback are not user history.
            messages = [m for m in state.get("messages", [])[:-1]
                        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)][-4:]
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        for m in list(messages):
            if isinstance(m, AIMessage):
                for call in m.tool_calls:
                    if call["id"] not in answered:
                        messages.append(ToolMessage(
                            content=dump({"success": False, "interrupted": True,
                                          "next": "检查已保存的任务进度与结果索引，从未完成位置继续，不能假定整次调用都未执行"}),
                            tool_call_id=call["id"], name=call["name"],
                        ))
        messages.append(HumanMessage(content=state["current_query"]))
        plan = copy.deepcopy(combined.get("task_plan", []))
        if approval_action == "resume":
            mark_chat_delivery(scope, plan)
        if reply_task_ready(scope, plan):
            messages = [HumanMessage(content=state["current_query"])]
        result = {
            "trip_state": trip.model_dump(mode="json"), "agent_messages": messages,
            "report_workspace": combined.get("report_workspace", {}),
            "task_plan": plan,
            "active_goal": combined.get("active_goal") or state["current_query"],
            "root_goal": combined.get("root_goal") or combined.get("report_workspace", {}).get("request") or combined.get("active_goal") or state["current_query"],
            "semantic_summary": combined.get("semantic_summary", {}),
            "selected_results": combined.get("selected_results", []),
            "request_scope": scope, "plan_approval": approval, "pending_plan": pending,
            "unpresented_plan": revision, "delivery_feedback": {},
            "export_source": export_source, "previous_run_status": combined.get("run_status", "completed"),
            "summary_version": combined.get("summary_version", 0),
            "loaded_tools": combined.get("loaded_tools", []),
            "loaded_skills": combined.get("loaded_skills", {}),
            "planner_round": 0, "turn_results": [], "tool_results": [], "pending_calls": [],
            "assistant_response": "", "agent_done": False, "final_checks": 0,
            "run_status": "running" if running else combined.get("run_status", "completed"),
            "phase": "agent", "repeat_reads": combined.get("repeat_reads", {}),
            "read_progress": combined.get("read_progress", {}),
            "protocol_repairs": 0,
            "action_observations": combined.get("action_observations", {}),
            "needs_action": False,
            "needs_progress": False,
            "stalled_batches": 0,
            "research_checkpoint": "",
            "initial_report_version": approval.get("base_report_version", base_version) if running else base_version,
            "checked_artifact_version": 0,
        }
        self.event(state, "agent.request_scope", scope)
        if not export_source:
            self.save({**state, **result}, trip)
        self.store.migrate_executions(state["thread_id"], self.registry.normalise_arguments)
        self.registry._legacy_imported.add(state["thread_id"])
        return result

    def model(self, state):
        self.gateway.usage_recorder = lambda record: self.event(state, "model.usage", record)
        current = dict(state)
        scope = state["request_scope"]
        feedback = state.get("delivery_feedback", {})
        if state.get("export_source"):
            return self.export_existing(state)
        last = (state.get("turn_results") or [{}])[-1]
        if last.get("error_type") == "approved_route_conflict":
            version = state["plan_approval"]["version"]
            issue = last["error"]
            self.event(state, "agent.approved_route_conflict", {"version": version, "issue": issue})
            return self.finish(state, {}, "", {"decision": "ask_user", "answer": (
                f"这次排出的路线还不符合你批准的 v{version}：{issue}。我没有把这条路线标为完成。"
                "建议先保留原定城市和停留时间，从其它可调整的行程里腾出时间。"
                "你希望保留原要求，还是愿意调整上面这些安排？"
            )})
        file_task = report_allowed(scope)
        if file_task and (delivery := self.check_ready_report(state)) is not None:
            return delivery
        if scope.get("reuse_lookup"):
            current["recent_lookup_quotes"] = recent_lookup_quotes(self.db, state["thread_id"])
        if state.get("stalled_batches", 0) >= 3:
            if any(r.get("error_type") == "missing_user_input" for r in state.get("turn_results", [])[-6:]):
                return self.finish(state, {}, "", {"decision": "ask_user", "answer": ORIGIN_QUESTION})
            workspace = state.get("report_workspace", {})
            gaps = list(workspace.get("active_issues", [])) if file_task else []
            if file_task and not gaps:
                gaps = [f"{edge['origin']} → {edge['destination']}尚缺可用交通资料"
                        for edge in workspace.get("data", {}).get("transport", []) if not edge.get("options")]
            errors = [r["error"] for r in state.get("turn_results", [])[-6:]
                      if not r.get("success") and r.get("error")
                      and r.get("name") not in {"inspect_tools", "load_skill"}]
            issue = "；".join(dict.fromkeys((gaps or errors)[0:3])) or "连续尝试没有得到新信息或完成新的步骤"
            return self.block(state, {}, issue)
        synthesize_reply = reply_task_ready(scope, state.get("task_plan", []))
        stalled = state.get("needs_progress", False)
        repeats = feedback.get("repeated_actions", [])
        stalled_reading = stalled and repeats and all(r["name"] in READ_TOOLS for r in repeats)
        synthesizing = (synthesize_reply or stalled_reading and not file_task) and not state.get("needs_action", False)
        if synthesizing:
            current["delivery_material"] = saved_material(self.store, state)
            current["delivery_feedback"] = {**feedback, "instruction": (
                "使用已保存结论和证据，给出本轮可交付的实际答复。"
                "若缺少决定性数据，具体指出对象、字段和为什么必要；不重复读取同一批预览。"
            )}
        if file_task and (stalled or feedback.get("decision") == "protocol"):
            current["loaded_tools"] = list(dict.fromkeys([*current.get("loaded_tools", []), "export_report"]))
            current["delivery_feedback"] = {**feedback, "instruction": (
                "当前页读完就推进工作。已有真实航班候选时，用select_result保存取舍，"
                "或用itinerary_step transport的result_refs把已查结果关联到工作区。"
                "根据工作区继续缺失步骤，只有完整后才调用export_report。"
            )}
        elif file_task and not state.get("report_workspace"):
            current["delivery_feedback"] = {**feedback, "instruction": (
                "这是完整报告。复用会话已保存的事实和检索结果，先核对用户实际确认的条件，"
                "使用update_trip补齐缺失的事实并用itinerary_step prepare建立工作区。"
                "已有聊天草稿可以帮助恢复约束，结构化日程、报价、来源必须用工作区证据建立。"
            )}
        trip = TripState.model_validate(state["trip_state"])
        if scope.get("requires_departure") and not permitted_origins(scope, trip):
            current["delivery_feedback"] = {"instruction": (
                "当前交付需要比较或安排从出发地出发的交通，但用户没有给出任何出发地。"
                "现在只需向用户问出发城市或机场，可给多个选项；不猜、不确认旧版、不自行做票价查询。"
            )}
        skills = self.skills
        current["loaded_skills"] = {
            n: skills[n]["path"].read_text(encoding="utf-8")
            for n in state.get("loaded_skills", {}) if n in skills
        }
        builtins = [self.tool_spec(name) for name in sorted(CORE_TOOLS)]
        definitions = builtins + [spec for name in current.get("loaded_tools", []) if name not in CORE_TOOLS
                                  and (spec := self.tool_spec(name))]
        inventory = [spec["function"] for spec in self.builtins.values()] + self.registry.capability_catalog()
        catalog = [{"name": x["name"], "description": x["description"][:160]} for x in inventory]
        fixed = conservative_tokens(dump(definitions)) + conservative_tokens(self.policy) + conservative_tokens(dump({
            "facts": trip_summary(trip, self.db.latest_report(trip.trip_id), include_details=not bool(state.get("report_workspace"))),
            "plan": state.get("task_plan", []), "skills": current["loaded_skills"], "catalog": catalog,
            "delivery_material": current.get("delivery_material", []),
            "recent_lookup_quotes": current.get("recent_lookup_quotes", []),
            "approved_plan": approved_plan_view(current),
        })) + 1200
        current = compact_context(current, self.gateway, self.db, fixed)
        current["_trip"] = trip
        try:
            messages, manifest = build_agent_messages(self.db, self.settings, current, self.policy, catalog, skills, definitions)
        except ContextCapacityError:
            # The recent groups themselves can exceed the remaining window after a
            # report or many schemas are added. Summarize them too instead of stopping.
            current = compact_context(current, self.gateway, self.db, fixed, force=True, keep_groups=0)
            try:
                messages, manifest = build_agent_messages(self.db, self.settings, current, self.policy, catalog, skills, definitions)
            except ContextCapacityError:
                current["loaded_tools"] = []
                definitions = builtins
                messages, manifest = build_agent_messages(self.db, self.settings, current, self.policy, catalog, skills, definitions)
        response = self.gateway.agent_turn(messages, definitions,
                                           final_only=synthesizing or not file_task and feedback.get("decision") == "revise",
                                           require_action=state.get("needs_action", False),
                                           output_tokens=3600 if synthesizing and scope["scenario"] == "execute" else 2200)
        round_number = state.get("planner_round", 0) + 1
        self.event(state, "agent.plan_decision", {
            "action": "use_tool" if response.tool_calls else "respond",
            "tool_name": ",".join(c["name"] for c in response.tool_calls) or None,
            "reason": "模型原生工具调用" if response.tool_calls else "候选用户答复",
            "loaded_tool_spec": bool(state.get("loaded_tools")), "planner_round": round_number,
        })
        changes = {"agent_messages": [*current.get("agent_messages", []), response],
                   "loaded_tools": current.get("loaded_tools", []),
                   "semantic_summary": current.get("semantic_summary", {}),
                   "summary_version": current.get("summary_version", 0),
                   "context_manifest": manifest, "planner_round": round_number}
        if response.tool_calls:
            changes["pending_calls"] = response.tool_calls
            self.save({**state, **changes})
            return changes
        answer = self.gateway._content_text(response).strip()
        if not answer or "<｜｜DSML｜｜" in answer or "<tool_call>" in answer:
            return self.repair_protocol(state, changes)
        verdict = self.check_delivery({**state, **changes, "delivery_material": current.get("delivery_material"),
                                       "recent_lookup_quotes": current.get("recent_lookup_quotes")}, answer)
        if verdict["decision"] == "research":
            marker = hashlib.sha256(canonical({
                "plan": state.get("task_plan"), "tools": state.get("loaded_tools"), "skills": list(state.get("loaded_skills", {})),
                "workspace": state.get("report_workspace"), "reads": state.get("read_progress"),
                "evidence": state.get("trip_state", {}).get("evidence_ids"), "selected": state.get("selected_results"),
            }).encode()).hexdigest()
            if marker == state.get("research_checkpoint"):
                return self.block(state, changes, verdict.get("issue") or "仍缺少可核实的结果")
            changes["research_checkpoint"] = marker
        decision = verdict["decision"]
        self.event(state, "agent.delivery_decision", {
            "scenario": scope["scenario"], "decision": decision, "issue": verdict.get("issue"),
            "requirement_index": verdict.get("requirement_index"),
        })
        replacement = verdict.get("answer")
        has_replacement = isinstance(replacement, str) and bool(replacement.strip())
        if decision == "research" or (decision == "revise" and not has_replacement):
            issue = str(verdict.get("issue") or "请完成本轮要求。")[:900]
            if decision == "revise" and state.get("final_checks", 0):
                return self.block(state, changes, issue)
            return {**changes, "pending_calls": [], "needs_action": decision == "research",
                    "final_checks": state.get("final_checks", 0) + int(decision == "revise"), "protocol_repairs": 0,
                    "delivery_feedback": {"decision": decision, "issue": issue}}

        return self.finish(state, changes, answer, verdict)

    def repair_protocol(self, state, changes):
        changes["agent_messages"].pop()
        if state.get("protocol_repairs", 0):
            return self.block(state, changes, "模型连续两次未返回可执行的工具调用或有效答复")
        self.event(state, "agent.protocol_repair", {"attempt": 1})
        return {**changes, "pending_calls": [], "needs_action": False,
                "protocol_repairs": 1, "needs_progress": False,
                "delivery_feedback": {"decision": "protocol", "issue": "请使用原生工具调用，或用自然语言直接答复。"}}

    def block(self, state, changes, issue):
        issue = str(issue)[:350].rstrip("。")
        self.event(state, "agent.execution_blocked", {"issue": issue})
        return self.finish(state, changes, "", {
            "decision": "blocked", "answer": f"本轮停在这里：{issue}。已保存完成的步骤和查询结果。",
        })

    def export_existing(self, state):
        row = self.db.latest_report(state["trip_id"])
        chat = state["export_source"]["kind"] in {"chat", "reply", "chat_file"}
        if delivered_report(row, state["initial_report_version"], allow_chat=chat):
            return self.finish(state, {}, "", {"decision": "pass"})
        if state.get("turn_results"):
            issue = state["turn_results"][-1].get("issue") or "导出的 HTML 或 Markdown 文件未通过检查"
            return self.block(state, {}, issue)
        call = {"id": f"export-{state['initial_report_version'] + 1}", "name": "export_report",
                "args": {"operation": "reexport"}}
        return {"pending_calls": [call], "loaded_tools": list(dict.fromkeys([*state["loaded_tools"], "export_report"])),
                "agent_messages": [*state["agent_messages"], AIMessage(content="", tool_calls=[call])]}

    def check_ready_report(self, state):
        workspace = state.get("report_workspace", {})
        if not workspace.get("rendered") or state.get("plan_approval", {}).get("status") != "approved":
            return None
        if route_issues(state["plan_approval"].get("route_contract"),
                        workspace.get("data", {}).get("region", {}).get("cities", [])):
            return None
        row = self.db.latest_report(state["trip_id"])
        if (not row or row["version"] == state.get("checked_artifact_version")
            or workspace.get("artifacts", {}).get("html_path") != row["html_path"]
            or not delivered_report(row, state["initial_report_version"])):
            return None
        plan = copy.deepcopy(state.get("task_plan", []))
        if any(task["status"] != "completed" for task in plan[:-1]):
            return None
        if plan and plan[-1]["status"] != "completed":
            last = plan[-1]
            legacy_delivery = not last.get("completion") and re.search(
                r"(?:报告|攻略|HTML|Markdown).*(?:渲染|导出|交付)|(?:渲染|导出|交付).*(?:报告|攻略|HTML|Markdown)",
                last["description"], re.IGNORECASE,
            )
            if last.get("completion") != "report" and not legacy_delivery:
                return None
            last.update(status="completed", completion="report", summary="标准报告已写入并通过文件与内容校验。",
                        result_refs=[f"artifact:{row['html_path']}"])
        answer = receipt(row)
        verdict = self.check_delivery({**state, "task_plan": plan}, answer)
        self.event(state, "agent.artifact_checked", {"version": row["version"], "decision": verdict["decision"],
                                                    "issue": verdict.get("issue")})
        if verdict["decision"] in {"pass", "revise"}:
            self.progress("plan", {"goal": state["active_goal"], "tasks": plan})
            return self.finish(state, {"task_plan": plan}, answer, {"decision": "pass"})
        return {"checked_artifact_version": row["version"],
                "needs_action": verdict["decision"] == "research",
                "delivery_feedback": verdict}

    def finish(self, state, changes, answer, verdict):
        decision = verdict["decision"]
        scope = state["request_scope"]
        replacement = verdict.get("answer")
        has_replacement = isinstance(replacement, str) and bool(replacement.strip())
        answer = replacement.strip() if has_replacement and decision != "pass" else answer
        approval = dict(state.get("plan_approval", {}))
        pending = dict(state.get("pending_plan", {}))
        revision = dict(state.get("unpresented_plan", {}))
        status = state["run_status"]
        if state.get("export_source"):
            if decision == "pass":
                answer = receipt(self.db.latest_report(state["trip_id"]))
            status = state["previous_run_status"]
        elif decision == "ask_user":
            status = "waiting_for_user" if approval.get("status") != "approved" or scope["scenario"] == "execute" else status
        elif decision == "blocked":
            status = "blocked" if scope["scenario"] == "execute" or not approval else status
        elif scope["scenario"] == "plan_review":
            if pending:
                self.store.archive(state["thread_id"], {k: state[k] for k in (
                    "pending_plan", "request_scope", "run_status",
                ) if k in state})
            version = max(approval.get("version", 0), pending.get("version", 0)) + 1
            answer = re.sub(r"^\s*\*\*待确认计划\s*v\d+[^\n]*?\*\*\s*", "", answer, count=1)
            answer = f"**待确认计划 v{version}**\n\n{answer}"
            pending = {"version": version, "status": "pending", "proposal": answer,
                        "scope": scope, "request": state["current_query"]}
            revision = {}
            if approval.get("status") != "approved":
                status = "waiting_for_user"
            self.event(state, "agent.plan_proposed", {"version": version})
        elif scope["scenario"] == "execute":
            approval["status"] = "completed"
            status = "completed"
            if report_allowed(scope):
                answer = receipt(self.db.latest_report(state["trip_id"]), verdict.get("delivery_note"))
            elif reply_task_ready(scope, state.get("task_plan", [])):
                delivery = ToolExecution(thread_id=state["thread_id"], tool_name="deliver_reply",
                    arguments={"plan_version": approval.get("version")},
                    result=ToolResult(success=True, tool_name="deliver_reply", source_type="internal",
                                      source_name="agent_runtime", data={"answer": answer}))
                self.db.save_tool_execution(delivery)
                plan = copy.deepcopy(state["task_plan"])
                plan[-1].update(status="completed", summary="完整答复已通过检查并发出。",
                                result_refs=[delivery.result.result_id])
                changes["task_plan"] = plan
                self.progress("plan", {"goal": state["active_goal"], "tasks": plan})
        elif approval.get("status") not in {"pending", "approved", "superseded"}:
            status = "completed"
        messages = list(changes.get("agent_messages", state.get("agent_messages", [])))
        if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
            messages.pop()
        if state.get("export_source"):
            messages = self.store.load(state["thread_id"]).get("agent_messages", [])
        else:
            messages.append(AIMessage(content=answer))
        changes["agent_messages"] = messages
        changes.update(assistant_response=answer, messages=[AIMessage(content=answer)],
                       pending_calls=[], agent_done=True, run_status=status, plan_approval=approval,
                       pending_plan=pending, unpresented_plan=revision, delivery_feedback={}, needs_action=False)
        if not state.get("export_source"):
            self.save({**state, **changes})
        self.event(state, "agent.turn_finished", {"decision": decision, "run_status": status})
        return changes

    def check_delivery(self, state, answer):
        scope = state["request_scope"]
        scenario = scope["scenario"]
        facts = {"current_request": state["current_query"]}
        if scenario == "lookup":
            facts["current_date"] = datetime.now(ZoneInfo("Asia/Hong_Kong")).date().isoformat()
        if state.get("recent_lookup_quotes"):
            facts["matching_previous_quotes"] = state["recent_lookup_quotes"]
        if scenario in {"compare", "lookup", "execute"}:
            facts["recent_evidence"] = [{"name": r.get("tool_name"), "success": r.get("success"),
                                         "data": selection_view(r["tool_name"], r.get("data", {}))
                                         if r.get("tool_name") in {"google_flights__search_flights", "amap_maps__maps_weather"}
                                         else tool_view(r.get("data", {}), limit=2)}
                                        for r in state.get("tool_results", [])[-3:]]
            facts["actions"] = state.get("turn_results", [])[-6:]
        if scenario in {"progress", "plan_review", "execute"}:
            facts["plan"] = state.get("task_plan", [])
            facts["run_status"] = state.get("run_status")
        if scenario == "plan_review" and (revision := state.get("unpresented_plan")):
            facts["unpresented_revision"] = revision.get("requests") or [revision["request"]]
        if scenario == "execute" and scope["artifact"] == "none":
            facts["delivery_material"] = state.get("delivery_material") or saved_material(self.store, state)
        if scenario in {"progress", "execute"}:
            facts["approval_status"] = state.get("plan_approval", {}).get("status")
        if approved := approved_plan_view(state):
            facts["approved_plan"] = approved
        row = None
        if report_allowed(scope) or scenario == "progress":
            row = self.db.latest_report(state["trip_id"])
            report = json.loads(row["payload_json"]).get("report", {}) if row else {}
            facts["report"] = {"version": row["version"], "html_path": row["html_path"],
                               "html_exists": Path(row["html_path"]).is_file(),
                               "markdown_exists": Path(row["markdown_path"]).is_file(),
                               "quality": report.get("quality")} if row else {}
            if report_allowed(scope):
                facts["version_before"] = state["initial_report_version"]
                facts["report"]["budget"] = report.get("budget")
                facts["report"]["days"] = [{"day": d["day"], "city": d.get("city"),
                                             "experiences": [b.get("name") for b in d.get("timeline", [])]}
                                            for d in report.get("days", [])]
                facts["report"]["transport"] = [{"origin": t.get("origin"), "destination": t.get("destination"),
                                                 "options": [{k: o.get(k) for k in ("name", "status", "price", "currency")}
                                                             for o in t.get("options", [])[:3]]}
                                                for t in report.get("transport", [])]
        if scenario == "plan_review" and scope["artifact"] != "none" and (
            "<!doctype" in answer.lower() or "<html" in answer.lower()
        ):
            return {"decision": "revise", "issue": "只给出待确认的步骤与文件交付方式，省略HTML源码。"}
        verdict = check_reply(self.gateway, scope, facts, answer)
        trip = TripState.model_validate(state["trip_state"])
        missing_origin = scope.get("requires_departure") and not permitted_origins(scope, trip)
        missing_origin = missing_origin or any(a.get("error_type") == "missing_user_input"
                                               for a in state.get("turn_results", []))
        if missing_origin and scenario in {"plan_review", "execute", "lookup", "compare"}:
            question = str(verdict.get("answer") or answer)
            if len(question) < 300 and "待确认计划 v" not in question and "出发" in question and re.search(r"[？?]", question):
                return {"decision": "ask_user", "answer": question}
            return {"decision": "ask_user", "answer": ORIGIN_QUESTION}
        if scenario != "execute":
            return verdict
        if report_allowed(scope):
            workspace_file = state.get("report_workspace", {}).get("artifacts", {}).get("html_path")
            matching = scope["artifact"] == "report_copy" or row and workspace_file == row["html_path"]
            if not matching or not delivered_report(row, facts["version_before"]):
                # A missing file is still actionable unless the user must
                # resolve a constraint or a recorded tool failure blocks it.
                missing_inputs = not (
                    trip.destination and trip.duration_days and trip.travel_dates.start_date
                )
                action_errors = {a.get("error_type") for a in state.get("turn_results", [])[-4:] if not a["success"]}
                question = str(verdict.get("answer") or answer)
                if (verdict["decision"] == "ask_user" and re.search(r"[？?]", question)
                    and (missing_inputs or "user_constraint_conflict" in action_errors)):
                    return verdict
                if verdict["decision"] == "blocked" and state.get("stalled_batches", 0):
                    return verdict
                return {"decision": "research", "issue": "尚无对应批准任务且校验通过的新报告文件。复用现有材料继续完成或报告实际阻塞。"}
        if verdict["decision"] not in {"pass", "revise"}:
            return verdict
        plan = state.get("task_plan", [])
        if not reply_task_ready(scope, plan) and any(task["status"] != "completed" for task in plan):
            return {"decision": "research", "issue": "检查已批准任务中尚未完成的计划项，使用真实结果引用更新已完成项。"}
        return verdict

    def execute(self, state):
        working = dict(state)
        working["report_workspace"] = copy.deepcopy(state.get("report_workspace", {}))
        working["agent_messages"] = list(state.get("agent_messages", []))
        working["turn_results"] = list(state.get("turn_results", []))
        working["tool_results"] = list(state.get("tool_results", []))
        working["loaded_tools"] = list(state.get("loaded_tools", []))
        working["loaded_skills"] = dict(state.get("loaded_skills", {}))
        working["repeat_reads"] = dict(state.get("repeat_reads", {}))
        working["read_progress"] = copy.deepcopy(state.get("read_progress", {}))
        working["action_observations"] = dict(state.get("action_observations", {}))
        stalled = []
        trip = TripState.model_validate(state["trip_state"])
        self.registry.progress_callback = self.progress
        calls = state.get("pending_calls", [])
        for call in calls:
            if call["name"].endswith("get-tickets"):
                call["args"] = {**call.get("args", {}), "format": "json"}
            elif call["name"] == "google_flights__search_flights":
                call["args"] = resolve_flight_dates(call.get("args", {}), state["current_query"], trip,
                                                    state["request_scope"]["scenario"] == "execute")
        parallel = {}
        if len(calls) > 1 and all(not self.action_error(working, trip, c["name"], c.get("args", {}))
                                 and c["name"] in working["loaded_tools"] and self.registry.has(c["name"])
                                 and "get_feed_detail" not in c["name"] for c in calls):
            self.registry._refresh = False
            parallel = self.registry.run_many(calls, state["thread_id"])
        for call in state.get("pending_calls", []):
            if self.registry.cancelled:
                from travel_agent.control import ExecutionCancelled
                raise ExecutionCancelled()
            started = time.perf_counter()
            name, args = call["name"], call.get("args", {})
            self.progress("action", {"name": name, "operation": args.get("operation")})
            try:
                if denied := self.action_error(working, trip, name, args):
                    result = denied
                elif name in self.builtins and (name in CORE_TOOLS or name in working["loaded_tools"]):
                    Draft202012Validator(self.tool_spec(name)["function"]["parameters"]).validate(args)
                    result = self.internal(working, trip, name, args)
                elif name in working["loaded_tools"] and self.registry.has(name):
                    if name.endswith("get-tickets"):
                        args = {"format": "json", "limitedNum": 8, **args}
                    if "get_feed_detail" in name:
                        from travel_agent.graph.agent_nodes import _xhs_token_for_feed
                        prior = [x["payload"]["result"] for x in self.db.list_tool_executions(state["thread_id"], limit=200)]
                        token = _xhs_token_for_feed(prior, args.get("feed_id"))
                        if token:
                            args = {**args, "xsec_token": token}
                    self.registry._refresh = False
                    actual = parallel.get(call["id"]) or self.registry.run(name, args, state["thread_id"])
                    working["tool_results"] = [*working["tool_results"][-5:], actual.model_dump(mode="json")]
                    if actual.success and actual.result_id not in trip.evidence_ids:
                        evidence = EvidenceItem(evidence_id=actual.result_id, source_name=actual.source_name or name,
                            content=actual.summary, structured_data=actual.data, source_type=actual.source_type,
                            source_url=actual.source_url, fetched_at=actual.fetched_at, valid_until=actual.valid_until)
                        self.db.save_evidence(evidence)
                        trip.evidence_ids.append(evidence.evidence_id)
                    result = {"success": actual.success, "result_id": actual.result_id, "evidence_id": actual.result_id,
                              "reused": actual.reused, "source": actual.source_name,
                              "fetched_at": actual.fetched_at.isoformat(), "data": selection_view(name, actual.data),
                              "prepared_view": name == "google_flights__search_flights",
                              "error_type": actual.error_type, "error": actual.error_message}
                else:
                    result = {"success": False, "error": "工具尚未加载，请先inspect_tools"}
            except Exception as exc:
                result = {"success": False, "error_type": type(exc).__name__, "error": str(exc)[:900],
                          "next": "先根据已保存成果修复或换方法，不重新研究已经完成的部分"}
            if name in self.builtins:
                execution = ToolExecution(thread_id=state["thread_id"], tool_name=name, arguments=args,
                    result=ToolResult(success=result.get("success", True), tool_name=name, data=copy.deepcopy(result),
                                      error_type=result.get("error_type") or ("operation_error" if result.get("success") is False else None),
                                      error_message=str(result.get("error") or result.get("next") or "") if result.get("success") is False else None,
                                      source_type="internal", source_name="agent_runtime"),
                    duration_ms=int((time.perf_counter() - started) * 1000))
                self.db.save_tool_execution(execution)
                if name not in READ_TOOLS | {"inspect_tools", "load_skill", "update_plan"} and "result_id" not in result:
                    result["result_id"] = execution.result.result_id
            view = result if name in READ_TOOLS else project(result, limit=8, text_limit=1300)
            working["agent_messages"].append(ToolMessage(content=dump(view),
                                                        tool_call_id=call["id"], name=name))
            # Detect a repeated action with unchanged business state, not a task-length limit.
            # Read pagination remains valid because each field/offset has a different key.
            action_key = fingerprint(name, args)
            business = {"trip": trip.model_dump(mode="json"), "plan": working.get("task_plan"),
                        "workspace": working.get("report_workspace"), "selected": working.get("selected_results")}
            digest = hashlib.sha256(canonical(business).encode()).hexdigest()
            observed = working["action_observations"].get(action_key)
            paginated_read = name in READ_TOOLS and result.get("result_id") and args.get("mode") != "review" and (
                result.get("returned") or result.get("pages")
            )
            repeated = bool(result.get("unchanged")) or (not paginated_read and observed == digest)
            if repeated:
                stalled.append({"name": name, "result_id": result.get("result_id"),
                                "error": result.get("error"), "unchanged": True})
            working["turn_results"].append({"name": name, "arguments": args, "success": result.get("success", True),
                                             "result_id": result.get("result_id"), "reused": result.get("reused", False),
                                             "issue": result.get("next") or result.get("error"),
                                             "error": result.get("error"),
                                             "error_type": result.get("error_type"),
                                             "new_information": result.get("success", True) and not (
                                                 repeated or result.get("already_loaded")
                                                 or name == "inspect_tools" and not result.get("newly_loaded")
                                             )})
            working["action_observations"][action_key] = digest
            if not state.get("export_source"):
                self.save(working, trip)
        feedback = {}
        if stalled:
            feedback = {"decision": "no_progress", "repeated_actions": stalled,
                        "issue": "重复动作没有新结果。用已取得的信息回答，或采用不同参数和真实来源继续本轮任务。"}
            self.event(working, "agent.no_progress", {"actions": stalled})
        recent = working["turn_results"][-len(calls):] if calls else []
        needs_progress = bool(stalled) and all(not r["new_information"] for r in recent)
        advanced = any(r["new_information"] for r in recent)
        rounds = 0 if advanced else state.get("stalled_batches", 0) + 1
        return {k: v for k, v in {**working, "pending_calls": [], "needs_action": False,
                                 "needs_progress": needs_progress, "delivery_feedback": feedback,
                                 "stalled_batches": rounds, "protocol_repairs": 0 if advanced else state.get("protocol_repairs", 0),
                                 "final_checks": 0 if advanced else state.get("final_checks", 0)}.items()
                if k not in {"messages", "current_query", "user_id", "trip_id", "thread_id"}}

    def internal(self, state, trip, name, args):
        if name == "inspect_tools":
            found = [n for n in dict.fromkeys(args["names"]) if self.tool_spec(n)]
            fresh = [n for n in found if n not in CORE_TOOLS and n not in state["loaded_tools"]]
            state["loaded_tools"].extend(dict.fromkeys(fresh))
            result = {"success": bool(found), "loaded": found, "newly_loaded": fresh,
                      "unavailable": [n for n in args["names"] if n not in found]}
            if found:
                result["next"] = "已找到的定义可用；调用时仍按本轮要求检查执行条件。"
            else:
                result.update(error_type="unknown_tool", error="目录中没有这些工具名称。",
                              next="只用capabilities中的准确名称；没有适合的工具就换方法或说明缺口。")
            return result
        if name == "load_skill":
            spec = self.skills.get(args["name"])
            if not spec:
                raise ValueError("技能名称不在目录中")
            root = spec["path"].parent.resolve()
            path = (root / args.get("resource", "SKILL.md")).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("参考文件必须位于技能目录内")
            if path == spec["path"].resolve():
                if args["name"] in state["loaded_skills"]:
                    return {"success": True, "name": args["name"], "already_loaded": True,
                            "next": "技能已加载，可立即使用。"}
                state["loaded_skills"][args["name"]] = True
                return {"success": True, "name": args["name"], "next": "技能正文进入下一次模型输入"}
            return {"success": True, "content": path.read_text()[:10000]}
        if name == "update_plan":
            old_tasks = {t["id"]: t for t in state.get("task_plan", [])}
            ids = [t["id"] for t in args["tasks"]]
            if len(ids) != len(set(ids)) or sum(t["status"] == "in_progress" for t in args["tasks"]) > 1:
                raise ValueError("任务ID必须唯一，最多一个任务in_progress")
            for task in args["tasks"]:
                if task.get("completion") == "reply":
                    if state["request_scope"]["artifact"] != "none" or task is not args["tasks"][-1]:
                        raise ValueError("reply只用于聊天交付的最后一项")
                    if task["status"] == "completed":
                        raise ValueError("最后一项由最终答复通过检查后自动完成，请直接回复用户")
                if task.get("completion") == "report" and (state["request_scope"]["artifact"] != "report" or task is not args["tasks"][-1]):
                    raise ValueError("report只用于已批准报告文件的最后交付项")
                if task["status"] == "completed" and not task.get("result_refs"):
                    raise ValueError(f"已完成任务 {task['id']} 需关联已保存结果、workspace:字段或artifact:实际文件路径")
                if task["status"] == "completed" and not str(task.get("summary", "")).strip():
                    old = old_tasks.get(task["id"], {})
                    if old.get("status") != "completed" or old.get("result_refs") != task.get("result_refs"):
                        raise ValueError(f"完成任务 {task['id']} 时请用summary保存供下一步复用的结论")
                    if old.get("summary"):
                        task["summary"] = old["summary"]
                for ref in task.get("result_refs", []):
                    if not self.reference_exists(state, ref):
                        raise ValueError(f"结果引用不存在：{ref}")
            mark_chat_delivery(state["request_scope"], args["tasks"])
            state["task_plan"] = args["tasks"]
            goal = state["active_goal"]
            self.save(state, trip)
            self.progress("plan", {"goal": goal, "tasks": args["tasks"]})
            return {"success": True, "tasks": args["tasks"], "goal": goal}
        if name == "select_result":
            found = self.store.result(state["thread_id"], args["result_id"])
            if not found or not found["result"].get("success"):
                raise ValueError("必须选择已取得的成功结果")
            data = found["result"].get("data", {})
            candidates = next((data[k] for k in ("flights", "properties", "hotels", "items", "results") if isinstance(data.get(k), list)), None)
            if candidates is not None and args.get("item_index", 0) >= len(candidates):
                raise ValueError("候选序号超出实际结果范围，请按真实候选从0开始选择")
            selection = {"result_id": args["result_id"], "item_index": args.get("item_index", 0), "purpose": args["purpose"]}
            state["selected_results"] = [s for s in state.get("selected_results", []) if s["purpose"] != args["purpose"]] + [selection]
            self.save(state, trip)
            return {"success": True, "selected": selection}
        if name in {"read_state", "read_result"}:
            result = self.read(state, trip, name, args)
            if args.get("mode") == "review" or result.get("exhausted") and not result.get("returned") and not result.get("pages"):
                return result
            if result.get("pages") or result.get("returned"):
                return remember_pages(state.setdefault("read_progress", {}), result)
            # Canonical result identity handles both evidence entry points.
            key = canonical((result["result_id"], result.get("field"))) if result.get("result_id") else name + canonical(args)
            digest = canonical(result)
            old = state["repeat_reads"].get(key, {})
            repeats = old.get("repeats", 0) + 1 if old.get("digest") == digest else 0
            state["repeat_reads"][key] = {"digest": digest, "repeats": repeats}
            if repeats:
                if result.get("success") is False:
                    return {**result, "unchanged": True}
                return {"success": True, "unchanged": True, "next": "与上次读取完全相同，复用已读内容或提供更具体的字段。"}
            return result
        if name == "refresh_tool":
            tool = args["name"]
            if tool not in state["loaded_tools"]:
                raise ValueError("刷新前请inspect_tools加载参数")
            self.registry._refresh = True
            try:
                result = self.registry.run(tool, args["arguments"], state["thread_id"])
            finally:
                self.registry._refresh = False
            self.event(state, "tool.explicit_refresh", {"name": tool, "reason": args["reason"]})
            return {"success": result.success, "result_id": result.result_id, "data": tool_view(result.data)}
        if name == "update_trip":
            from travel_agent.graph.agent_nodes import _apply_requirements
            req = args["requirements"]
            scope = state["request_scope"]
            if req.get("origin") and "departure_origins" in scope and req["origin"].strip() not in scope["departure_origins"]:
                return {"success": False, "error_type": "unconfirmed_origin",
                        "error": "没有用户原话支持这个出发地，未保存。",
                        "next": "只用用户给出的出发地原名；没有出发地时追问。"}
            if state["request_scope"]["scenario"] == "compare":
                choices = {"destination", "duration_days", "start_date", "end_date", "confirmed_decisions"}
                req = {k: v for k, v in req.items() if k not in choices}
            changed = _apply_requirements(trip, {"requirements": req})
            trip.__dict__.update(changed.__dict__)
            if trip.travel_dates.start_date and trip.travel_dates.end_date and not trip.duration_days:
                trip.duration_days = (trip.travel_dates.end_date - trip.travel_dates.start_date).days + 1
            for key in ("confirmed_decisions", "rejected_options"):
                if req.get(key):
                    setattr(trip, key, list(dict.fromkeys([*getattr(trip, key), *req[key]])))
            self.save(state, trip)
            result = {"success": True, "facts": trip_summary(trip)}
            if trip.travel_dates.start_date and trip.travel_dates.end_date and trip.duration_days:
                actual_days = (trip.travel_dates.end_date - trip.travel_dates.start_date).days + 1
                if actual_days != trip.duration_days:
                    result["conflict"] = {"calendar_days": actual_days, "requested_days": trip.duration_days}
            return result
        if name == "remember":
            self.db.save_memory(MemoryItem(user_id=trip.user_id, content=args["content"], kind=args["kind"], source="user"))
            return {"success": True, "remembered": args["content"]}
        if name in {"itinerary_step", "export_report"}:
            if name == "export_report" and state["request_scope"]["artifact"] == "report_copy":
                if args["operation"] != "reexport":
                    raise ValueError("只重导出时使用reexport，保留已保存报告的内容")
                source = state.get("export_source", {})
                row = self.db.latest_report(trip.trip_id)
                chat = source.get("kind") in {"chat", "reply", "chat_file"}
                if delivered_report(row, state["initial_report_version"], allow_chat=chat):
                    # A restart after the file was committed must not export again.
                    return {"success": True, "html_path": row["html_path"], "markdown_path": row["markdown_path"],
                            "version": row["version"], "reused": True}
                if source.get("kind") in {"chat", "reply"}:
                    markdown = source.get("markdown")
                    if source["kind"] == "reply":
                        result = self.store.result(state["thread_id"], source["result_id"])
                        markdown = (result or {}).get("result", {}).get("data", {}).get("answer")
                    if not markdown:
                        return {"success": False, "error": "找不到所选行程的已保存正文。"}
                    heading = re.search(r"^#\s+(.+)", markdown, re.MULTILINE)
                    title = heading.group(1) if heading else f"{trip.destination or ''}旅行行程"
                    plan = TravelPlan(trip_id=trip.trip_id, title=title, destination=trip.destination,
                                      summary="", report={"chat_markdown": markdown})
                else:
                    if not row or source and row["version"] != source["version"]:
                        return {"success": False, "error": "找不到指定版本的已保存报告。"}
                    plan = TravelPlan.model_validate_json(row["payload_json"])
                    if plan.report.get("chat_markdown") and not chat:
                        return {"success": False, "error": "现有文件是聊天导出，需要完整报告工作区。"}
                markdown, html = self.renderer.render(plan)
                version = self.db.latest_report(trip.trip_id)["version"]
                self.event(state, "report.exported", {"version": version, "html_path": str(html),
                                                     "source": source.get("kind", "report")})
                return {"success": True, "html_path": str(html), "markdown_path": str(markdown),
                        "version": version, "quality": plan.report.get("quality")}
            if args["operation"] == "reexport":
                raise ValueError("原样重导出只用于用户明确批准的导出任务")
            contract = None
            if args["operation"] != "prepare" and state.get("report_workspace") and (approved := approved_plan_view(state)):
                contract = state["plan_approval"].get("route_contract")
                if not contract or contract.get("schema") != 2 or contract["version"] != approved["version"]:
                    contract = extract_route_contract(self.gateway, approved, state["request_scope"].get("requirements", []))
                    state["plan_approval"] = {**state["plan_approval"], "route_contract": contract}
                    self.save(state)
            if args["operation"] == "render" and state.get("report_workspace", {}).get("rendered"):
                artifact = state["report_workspace"].get("artifacts", {})
                row = self.db.latest_report(trip.trip_id)
                if (row and artifact.get("html_path") == row["html_path"]
                    and delivered_report(row, state.get("initial_report_version", 0))
                    and not route_issues(contract, state["report_workspace"]["data"].get("region", {}).get("cities", []))):
                    return {**artifact, "version": row["version"], "reused": True, "unchanged": True}
            def checkpoint(workspace):
                state["report_workspace"] = workspace
                self.save(state, trip)
                self.progress("report", {"completed": workspace.get("completed"),
                                         "research_progress": workspace.get("research_progress", {})})
            result, workspace = report_operation(self.db, self.registry, self.gateway, self.renderer, trip,
                state.get("report_workspace", {}), args, checkpoint=checkpoint, route_contract=contract)
            state["report_workspace"] = workspace
            return result
        raise ValueError("未知内部能力")

    def reference_exists(self, state, reference):
        if reference.startswith("workspace:"):
            return bool(state.get("report_workspace", {}).get("data", {}).get(reference.split(":", 1)[1]))
        if reference.startswith("artifact:"):
            return Path(reference.split(":", 1)[1]).is_file()
        return bool(self.store.result(state["thread_id"], reference))

    def read(self, state, trip, name, args):
        section = args.get("section")
        offset, limit = args.get("offset", 0), min(args.get("limit", 3), 3)
        if name == "read_result":
            found = self.store.result(state["thread_id"], args["result_id"])
            if not found:
                return {"success": False, "next": "结果ID不存在；read_state results查看准确ID"}
            visited = set()
            while found["tool_name"] in {"read_result", "read_state"}:
                original = found.get("arguments", {}).get("result_id") or found.get("arguments", {}).get("target")
                if not original or original in visited:
                    break
                visited.add(original)
                source = self.store.result(state["thread_id"], original)
                if not source:
                    break
                found = source
            root = found["result"].get("data", {})
            result_id = found["result"].get("result_id") or found.get("execution_id") or args["result_id"]
            metadata = {"result_id": result_id, "tool": found["tool_name"], "arguments": found.get("arguments"),
                        "fetched_at": found["result"].get("fetched_at"), "prepared_view": True}
            value = root
            field = args.get("field", "")
            if args.get("mode") != "review":
                progress = state.get("read_progress", {}).get(result_id, {})
                canonical_field = field if field in progress or field.split(".")[0] in root else field.removeprefix("data.")
                field, entry = page_target(progress, canonical_field)
                if entry:
                    offset = first_unread(entry["ranges"])
                    if offset >= entry["total"]:
                        return exhausted_page(result_id, field, entry["total"])
                elif progress and not field:
                    cursors = [p["next_args"] for p in progress.values() if p.get("next_args")]
                    if cursors:
                        return {**metadata, "next_options": cursors,
                                "next": "有多个可读字段；选择一个字段名称继续。"}
                    return {**exhausted_page(result_id, "", sum(p["total"] for p in progress.values())),
                            "available_fields": list(root)}
            if field:
                paths = [field]
                if field.startswith("data."):
                    paths.append(field[5:])
                for path in paths:
                    value = root
                    for part in path.split("."):
                        if isinstance(value, dict):
                            value = value.get(part)
                        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                            value = value[int(part)]
                        else:
                            value = None
                    if value is not None:
                        break
                if value is None:
                    error = {"success": False, "available_fields": list(root),
                             "error": "字段路径不存在。field从原始数据根节点开始，预览外层和剩余条数不是字段。"}
                    cursors = []
                    for key, items in root.items():
                        if isinstance(items, list) and items:
                            progress = state.get("read_progress", {}).get(result_id, {}).get(key, {})
                            cursor = first_unread(progress.get("ranges", []))
                            if cursor < len(items):
                                cursors.append(next_args(result_id, key, cursor))
                    if len(cursors) == 1:
                        error["next_args"] = cursors[0]
                    elif cursors:
                        error["next_options"] = cursors
                    return error
                field = path
                metadata["field"] = field
                if args.get("mode") == "review":
                    metadata["review"] = True
                if isinstance(value, list) and any(isinstance(item, dict) and "remaining_items" in item for item in value):
                    return {"success": False, "truncated_snapshot": True,
                            "next": f"这条旧内部记录只存了预览，不能把remaining_items视为真实记录。请read_state section={field.split('.')[0]}读取当前工作区的完整保存结果，并指定城市/天数/ID或按实际next_offset翻页。"}
            if not field:
                return {**metadata, **preview_result(root, result_id)}
        elif section == "results":
            items = self.store.index(state["thread_id"], offset=offset, limit=limit + 1, tool_name=args.get("target"))
            return {"items": items[:limit], "next_offset": offset + limit if len(items) > limit else None}
        elif section == "plan":
            return {"goal": state.get("active_goal"), "tasks": state.get("task_plan", []), "status": state.get("run_status")}
        elif section == "summary":
            w = state.get("report_workspace", {})
            return {"facts": trip_summary(trip, self.db.latest_report(trip.trip_id), include_details=not bool(w)), "tasks": state.get("task_plan", []),
                    "workspace": {k: w.get(k) for k in ("completed", "research_progress", "active_issues", "rendered")},
                    "next": "使用已保存结论，按当前请求的交付方式完成任务。"}
        else:
            row = self.db.latest_report(trip.trip_id)
            data = state.get("report_workspace", {}).get("data") or (json.loads(row["payload_json"])["report"] if row else {})
            metadata = {"section": section, "prepared_view": True}
            if section == "evidence":
                value = [{"evidence_id": e.evidence_id, "source": e.source_name, "summary": e.content[:160],
                          "fields": list(e.structured_data)} for e in self.db.get_evidence(trip.evidence_ids)]
            else:
                value = data.get(section, [])
            target = args.get("target", "").casefold()
            if section == "days" and target.isdigit():
                value = [d for d in value if d.get("day") == int(target)]
                target = ""
            if section == "region" and target and isinstance(value, dict):
                value = [c for c in value.get("cities", []) if target in dump(c).casefold()]
            if target and isinstance(value, list):
                value = [x for x in value if target in dump(x).casefold()]
            if section == "evidence" and target and len(value) == 1:
                return self.read(state, trip, "read_result", {
                    "result_id": value[0]["evidence_id"],
                    **{k: args[k] for k in ("field", "offset", "limit", "mode") if k in args},
                })
            if section == "days":
                value = [{**{k: d.get(k) for k in ("day", "date", "city", "city_id", "stay_city", "type", "no_hotel", "timing_warning", "exception_reason", "available_from", "available_until")},
                          "timeline": [{**{k: b.get(k) for k in ("id", "type", "name", "city", "city_id", "start_time", "end_time", "duration_minutes", "status")},
                                        "alternatives": [{"id": a.get("id"), "name": a.get("name")} for a in b.get("alternatives", []) if isinstance(a, dict)]}
                                       for b in d.get("timeline", [])]}
                         for d in value]
                metadata["view"] = "完整时间线索引；点位详情用read_state places按ID读取，原始报告完整保存在工作区"
            elif section == "transport":
                value = [{**transfer_index(t), **{k: t.get(k) for k in ("type", "status")},
                          "options": [{k: o.get(k) for k in (
                              "name", "mode", "status", "baseline", "price", "currency", "price_basis", "per_person_price", "group_estimated_price", "quote_passengers",
                              "duration_minutes", "planning_duration_minutes", "departure_date", "departure_time", "arrival_date", "arrival_time",
                              "planning_departure_date", "planning_departure_time", "planning_arrival_date", "planning_arrival_time",
                          ) if o.get(k) is not None} for o in t.get("options", [])]}
                         for t in value]
                for item in value:
                    for option in item["options"]:
                        for key in ("departure_date", "departure_time", "arrival_date", "arrival_time"):
                            option[key] = option.pop("planning_" + key, None) or option.get(key) or (item["date"] if key == "departure_date" else None)
                metadata["view"] = "交通候选与计划时刻索引；原始航段和来源正文按来源result_id读取"
        if isinstance(value, list):
            items = []
            for item in value[offset:offset + limit]:
                view = project(item, limit=12, text_limit=2000)
                if items and len(dump({**metadata, "items": [*items, view]})) > 6000:
                    break
                if not items and len(dump(view)) > 5000:
                    view = tool_view(item, limit=3)
                    metadata["preview_only"] = True
                    metadata["next"] = "该项含大字段；用field定位列表下标及必要字段，如flights.0.legs，继续分页读取"
                items.append(view)
            if name == "read_result":
                return {**metadata, "items": items, **page_info(result_id, field, offset, len(items), len(value))}
            next_offset = offset + len(items)
            return {**metadata, "total": len(value), "items": items,
                    "next_offset": next_offset if next_offset < len(value) else None}
        if isinstance(value, str):
            content = value[offset:offset + 2000]
            if name == "read_result":
                return {**metadata, "text": content, **page_info(result_id, field, offset, len(content), len(value))}
            return {**metadata, "text": content, "next_offset": offset + len(content) if content and offset + len(content) < len(value) else None}
        return {**metadata, "data": project(value, limit=8, text_limit=2000)}
