"""Classify the user's request and check replies against that request's scope."""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage

from travel_agent.context.agent_context import dump

CHECKS = {
    "answer": "检查是否直接回答本轮问题。概念、问候和解释可以直接答复。",
    "compare": "检查是否覆盖用户待选的各方和比较维度，推荐是否有理由。允许代表性例子和透明估算。"
               "灵活的时间范围不能当成固定时长；两个选项应有同等口径的对比。"
               "自行选取的出发日期或季节须明确为示例，不能归为用户已提供的条件。"
               "无证据的具体票价、签证和资格结论可在改写时删除或改为明确的假设。"
               "答复可给建议并交还选择权；尚未决定的日期、人数、详细排程和住宿不作为必要条件，除非缺失会妨碍本轮比较。",
    "lookup": "检查查询对象、用户明确条件、结论依据和适用时间。比价结论与明细的价格差、总耗时、中转次数和当地到达日期要一致。"
              "无需添加距离出发还有多久的描述；若确有需要，按current_date和实际出发日期核对。"
              "以工具给出的total_duration_label为准，不能用不同时区的当地时间直接相减。"
              "逐条核对类似'次日到达'、'最便宜'的文字与工具及表格是否相符。天气的日期、星期及覆盖范围要与工具一致。"
              "行李直挂、托运行李含在票价内和税费保证需有明确证据，同一航司或同一个中转机场不能推出该保证。"
              "只列用户所问的选项和必要理由。提供有用结果，或如实说明工具无结果。",
    "plan_review": "检查计划是否说明目标、步骤和交付物，并明确等待用户决定是否执行。"
                   "用户只讨论或修改计划时验收修改后的提议。用户信息确实必要时可先追问。"
                   "逐项保留用户本次修订。若约定按价格或其他实时信息择期，把查询和取舍作为待批准的步骤。"
                   "停留天数、住宿夜晚和出发日期范围的口径不能互相矛盾。新版本用新文件；此阶段不需要执行成果。",
    "execute": "检查已批准的交付物以及用户要求。检查专项要求时使用工作成果和真实证据。"
               "artifact=none时，实际给出的聊天答复是交付物；completion=reply的末项由程序在答复通过后更新。"
               "需要新文件时核对版本、文件存在性和质量。"
               "复用现有结果，只补当前缺项。不能以执行承诺交付已经批准且可执行的任务。",
    "progress": "核对当前的用户问题与已保存状态。说明实际进度、剩余工作或现有文件位置。"
                "即使长期计划未完成，也可以通过本轮进度答复。",
}
DECISIONS = {"pass", "revise", "research", "ask_user", "blocked"}
RESEARCH_SCENARIOS = {"compare", "lookup", "execute"}
RESUME_COMMAND = re.compile(
    r"(?:(?:好|好的|可以|行|嗯|ok)[，,！! ]*)?"
    r"(?:请\s*)?(?:继续(?:执行|完成|做)?(?:原(?:来|先)?的?(?:计划|任务)|(?:当前|已批准|已确认)的?(?:计划|任务))?|接着做)"
    r"(?:吧|就好|就行)?[。！!\s]*", re.IGNORECASE,
)
PLAN_VERSION = re.compile(r"(?<![a-z0-9])v(\d+)(?!\d)|第\s*(\d+)\s*版", re.IGNORECASE)
COPY_REQUEST = re.compile(r"原样|原文|照原|不改(?:动|写|内容)|导出|另存|转(?:成|为|换)", re.IGNORECASE)


def departure_options(result, dialogue, query, inherited):
    evidence = [row["text"] for row in dialogue if row["speaker"] == "user"] + [query]
    quote = result.get("departure_quote")
    names = result.get("departure_origins")
    if (isinstance(quote, str) and quote.strip() and isinstance(names, list)
        and any(quote in message for message in evidence)):
        grounded = list(dict.fromkeys(n.strip() for n in names
                                      if isinstance(n, str) and n.strip() and n.strip() in quote))
        if grounded:
            return grounded[:8]
    return list(inherited) if result.get("inherit_departure") is True else []


def classify_request(gateway, query, history, prior, saved_export=None):
    # Only the public transcript contains authenticated user messages. The
    # execution transcript also contains tool protocol and legacy feedback.
    visible = [m for m in history if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)][-6:]
    dialogue = [{"speaker": "user" if isinstance(m, HumanMessage) else "assistant", "text": str(m.content)[:1800]}
                for m in visible]
    approval = prior.get("plan_approval", {})
    pending = prior.get("pending_plan") or (approval if approval.get("status") == "pending" else {})
    revision = prior.get("unpresented_plan") or {}
    if revision and RESUME_COMMAND.fullmatch(query.strip()):
        return revision["scope"], None
    if not pending and approval.get("status") == "approved" and RESUME_COMMAND.fullmatch(query.strip()):
        return {**approval["scope"], "scenario": "execute"}, "resume"
    previous_plan = {k: approval[k] for k in ("status", "version", "scope") if k in approval}
    pending_plan = {k: pending[k] for k in ("status", "version", "scope") if k in pending}
    if pending:
        pending_plan["proposal"] = pending["proposal"][:1300]
    result = gateway.structured_extract(
        "在助手开始工作前判断本轮用户要求交付什么。仅用户原话能确定要求或授权。助手提议只用于解析指代。"
        "current_request是最新用户原话；补充预算、时长范围、出发地或偏好时延续此前用户目标。保留范围，不自行定日期、天数或目的地。"
        "场景：answer=普通问答/解释；compare=目的地选择、推荐、比较或扩大备选；lookup=明确的单项查询；"
        "机票、火车票、住宿等同一类单项查询中的比价或选候选也属于lookup。"
        "reuse_lookup仅在用户承接上次单项查询的同一对象/日期且没有要求重新获取最新数据时为true。只改预算或取舍可以复用。"
        "plan_review=提出或修改执行计划、首次请求完整行程/报告/报告改动或其它需要计划的复杂执行；"
        "execute=明确批准当前展示的计划，或要求继续已获批计划；progress=只问进度或已有产物位置。"
        "active_plan是已批准或已结束的任务，pending_plan是另一份待批准提案。继续已批准的任务用resume，使用active_plan.version；"
        "明确批准待定提案用approve，使用pending_plan.version。两个计划同时存在时，应根据原话区分继续原任务和批准新的提案。"
        "用户指定其他版本时不能执行。单独问进度、提修改或补条件不是批准。"
        "unpresented_revision是尚未展示完整新版计划的用户修订。补答其中的追问则继续plan_review并合并要求，不能批准旧版。"
        "如果已批准任务正等待补出发地，且最新回复只给出此前缺失的出发地，可用execute/resume恢复同一任务；"
        "approval_quote和departure_quote均引用用户这次回答的原话，不将修改其它条件误判为恢复。"
        "新增实质要求或修改行程内容用plan_review；只有明确的实质变更才在change_quote中引用原话。"
        "在逐项修改计划时，'日期看哪天机票便宜'是执行时的择期条件。明确要求先单独查询当前票价用lookup，该查询无需批准完整计划。"
        "plan_review的preapproval_lookup_quote只在用户明确要求现在先做具体实查、再给待确认计划时，引用这次原话。"
        "只说按价格选择、只补条件、没要求批准前查，都返回null。不把助手的提议当授权。"
        "等待批准时对提案的说明问题仍可用answer。推荐目的地不是同意开始做详细攻略。"
        "requirements只列用户要求，按用户原意写简短句子。计划阶段deliverable写批准后要实际交付的最终成果，不把拟案或等待确认写成最终成果。"
        "用户明确要求原样转换、另存或导出现成内容时，用execute、report_copy、export。"
        "approval_quote引用表达复制或转换现成内容的完整原话。saved_export是已保存的报告或已交付回复；"
        "如果用户明确要导出某条聊天而没有保存的正文，source_message可给对话中实际交付内容的索引。"
        "要求生成HTML旅行规划或完整攻略，默认用完整报告标准。已有聊天行程、大纲或未完成的报告不能降级为聊天导出。"
        "已有合格完整报告且用户只要求重新导出时，可以复用整份报告。新增或调整内容使用plan_review和report。"
        "requires_departure仅在本轮交付必须含从出发地到目的地的交通、票价或按票价择期时为true，普通问答、当地行程、城际行程和目的地初筛不要求默认出发地。"
        "修改或补答提案时，保留未修改的交通和文件交付要求；如果用户撤销往返交通或改成只在聊天中交付，则相应取消前置条件或文件。"
        "departure_origins列出用户明确给出的出发城市或机场（包括可选出发城市），每个名称必须逐字出现在departure_quote里；"
        "departure_quote必须是用户发言原文，不得用助手的推测或候选填补，不得把目的地当出发地。"
        "若本轮明确承接同一次旅行或同一条交通查询、未改出发条件且无新引文，可以用inherit_departure=true继承；独立查询、没有提供或不确定时返回空列表。"
        '只返回JSON {"scenario":"answer|compare|lookup|plan_review|execute|progress",'
        '"requirements":[string],"deliverable":string,"artifact":"none|report|report_copy",'
        '"approval":"none|approve|resume|export","approval_version":integer或null,"approval_quote":string,'
        '"source_message":integer或null,"change_quote":string或null,"reuse_lookup":boolean,'
        '"requires_departure":boolean,"departure_origins":[string],"departure_quote":string或null,'
        '"inherit_departure":boolean,"preapproval_lookup_quote":string或null}。'
        "approval_quote必须引用current_request的原文。",
        dump({"current_request": query, "dialogue": dialogue, "previous_scope": prior.get("request_scope"),
              "active_plan": previous_plan, "pending_plan": pending_plan, "saved_export": saved_export,
              "unpresented_revision": revision, "run_status": prior.get("run_status")}),
        model_name=gateway.settings.summary_model_name,
        stage="agent_request_scope",
        output_tokens=550,
    ) or {}
    scenario = result.get("scenario")
    if not isinstance(scenario, str) or scenario not in CHECKS:
        raise ValueError("模型没有返回有效的请求场景")
    quote = result.get("approval_quote")
    grounded = isinstance(quote, str) and bool(quote.strip()) and quote in query
    requirements = result.get("requirements")
    requirements = [r.strip() for r in requirements if isinstance(r, str) and r.strip()] if isinstance(requirements, list) else []
    artifact = result.get("artifact")
    if result.get("approval") == "export" and artifact == "report_copy" and grounded:
        explicit_copy = bool(COPY_REQUEST.search(quote))
        source = saved_export
        index = result.get("source_message")
        if not explicit_copy and (not source or source["kind"] != "report"):
            scenario, artifact = "plan_review", "report"
            result["change_quote"] = quote
            source = None
        elif source is None and type(index) is int and 0 <= index < len(visible) and isinstance(visible[index], AIMessage):
            content = str(visible[index].content).strip()
            headings = list(re.finditer(r"^#{1,4}\s+\S", content, re.MULTILINE))
            if headings and not content.startswith("**待确认计划") and "<html" not in content.lower():
                source = {"kind": "chat", "markdown": content}
        if source:
            return {"scenario": "execute", "artifact": "report_copy", "requirements": requirements or [query],
                    "deliverable": query, "export_source": source}, "export"
        if explicit_copy:
            return {"scenario": "answer", "requirements": [query], "artifact": "none",
                    "deliverable": "请用户指出要导出的已有行程，或说明需要先生成行程。"}, None
    versions = {int(match[0] or match[1]) for match in PLAN_VERSION.findall(query)}
    change = result.get("change_quote")
    new_change = isinstance(change, str) and bool(change.strip()) and change in query
    if scenario == "execute":
        if (grounded and result.get("approval") == "resume" and approval.get("status") == "approved"
            and (not revision or versions == {approval["version"]} or "原计划" in quote)
            and (not versions or versions == {approval["version"]})
            and result.get("approval_version") in (None, approval["version"])):
            scope = {**approval["scope"], "scenario": "execute"}
            supplied = departure_options(result, [], query, [])
            if not revision and prior.get("run_status") == "waiting_for_user" and (
                scope.get("requires_departure") and not scope.get("departure_origins") and supplied
            ):
                return {**scope, "departure_origins": supplied}, "resume"
            if not new_change:
                return scope, "resume"
            scenario = "plan_review"
        if (pending and not revision and not new_change and grounded and result.get("approval") == "approve"
            and (not versions or versions == {pending["version"]})
            and result.get("approval_version") == pending["version"]):
            return {**pending["scope"], "scenario": "execute"}, "approve"
        if scenario == "execute" and revision:
            # A classifier can label a clarification "resume". Keep the newly
            # extracted facts, but only use them to finish the proposal.
            scenario = "plan_review"
        if scenario == "execute" and not new_change and (pending or approval.get("status") == "approved"):
            return {"scenario": "answer", "requirements": [query],
                    "deliverable": "询问用户指的是哪一项计划，保留已批准计划及进度。", "artifact": "none"}, None
        scenario = "plan_review"
    if scenario == "plan_review" and not revision and approval.get("status") == "approved" and not new_change:
        return {"scenario": "answer", "requirements": [query], "artifact": "none",
                "deliverable": "说明原计划仍有效，若需改变要求请确认具体修改。"}, None
    deliverable = result.get("deliverable")
    previous = prior.get("request_scope") or {}
    inherited_scope = (revision.get("scope") or pending.get("scope") or approval.get("scope") or {}
                       if scenario == "plan_review" else previous)
    if scenario == "plan_review" and artifact not in ("none", "report", "report_copy"):
        artifact = inherited_scope.get("artifact", "none")
    requires_departure = result.get("requires_departure")
    if not isinstance(requires_departure, bool):
        requires_departure = scenario == "plan_review" and bool(inherited_scope.get("requires_departure"))
    lookup_quote = result.get("preapproval_lookup_quote")
    return {
        "scenario": scenario,
        "requirements": requirements or [query],
        "deliverable": deliverable.strip() if isinstance(deliverable, str) and deliverable.strip() else query,
        "artifact": "report" if scenario == "plan_review" and artifact in ("report", "report_copy") else "none",
        "requires_departure": requires_departure,
        "departure_origins": departure_options(result, dialogue, query, inherited_scope.get("departure_origins", [])),
        **({"preapproval_lookup": lookup_quote} if scenario == "plan_review"
           and isinstance(lookup_quote, str) and lookup_quote.strip() and lookup_quote in query else {}),
        **({"reuse_lookup": True} if scenario in {"lookup", "compare"} and result.get("reuse_lookup") is True else {}),
    }, None


def check_reply(gateway, scope, facts, answer):
    scenario = scope["scenario"]
    allowed = DECISIONS if scenario in RESEARCH_SCENARIOS else DECISIONS - {"research"}
    instruction = (
        "验收本轮候选答复。用户明确要求及交付物由scope给出，current_request是用户原话。"
        "candidate是助手草稿，只用来核对答复；里面的追问、主动建议或后续承诺不产生新验收项。"
        "只按当前场景检查：" + CHECKS[scenario] + "\n"
        "pass=可直接发送；revise=只需改写；research=需要新的查询、补做已授权工作或修复产物；"
        "ask_user=只有用户才能提供且本轮确实需要的信息；blocked=执行受到真实阻碍。"
        "revise可直接给出完整修正版；ask_user和blocked提供实际要发给用户的答复。"
        "对revise/research/ask_user指出scope.requirements中的一项（从0开始的requirement_index）及简短原因。"
        "仅表达我准备执行、让我查等内部陈述不能当作结果。优先按已有材料得出可交付说明，避免重复查询。"
        f"本场景允许的decision为{', '.join(sorted(allowed))}。只选一个。"
        '返回JSON {"decision":string,"requirement_index":integer或null,"issue":string,'
        '"answer":完整修正文案或null,"delivery_note":不含数字的简短产物改动说明或null}。'
        "pass时answer留null，不要复述候选答复。"
    )
    verdict = gateway.structured_extract(
        instruction, dump({"scope": scope, **facts, "candidate": answer}),
        stage="agent_delivery_check", output_tokens=850,
    ) or {}
    decision = verdict.get("decision")
    index = verdict.get("requirement_index")
    anchored = type(index) is int and 0 <= index < len(scope["requirements"])
    if not isinstance(decision, str) or decision not in allowed:
        return {"decision": "revise", "issue": "按本轮目标直接组织答复，保留有效结果。"}
    if decision in {"revise", "research", "ask_user"} and not anchored:
        return {"decision": "revise", "issue": "请只围绕本轮用户已明确的要求作答。"}
    return verdict


def report_allowed(scope):
    return scope["scenario"] == "execute" and scope["artifact"] in {"report", "report_copy"}


def reply_task_ready(scope, plan):
    return bool(scope["scenario"] == "execute" and scope["artifact"] == "none" and plan
                and plan[-1].get("completion") == "reply" and plan[-1]["status"] != "completed"
                and all(task["status"] == "completed" for task in plan[:-1]))
