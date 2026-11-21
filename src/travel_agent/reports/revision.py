"""Resolve natural-language edits against stable report objects and dependencies."""

from __future__ import annotations

import copy

SECTIONS = {"route", "days", "hotel_regions", "transport", "food", "budget", "notes", "insights"}


def revision_scope(planner, previous):
    planner.change_request = {}
    if not previous or previous.get("schema_version") != 2:
        return set(SECTIONS), "首次研究完整行程"
    before, after = previous["trip"], planner.trip.model_dump(mode="json")
    decision = planner.ask(
        "把用户的自然语言修改拆成可验收的任务，只选直接受影响部分。"
        "sections可用route城市顺序/停留天数、days每日游览、hotel_regions住宿、transport交通、food吃喝商店活动、"
        "budget费用、notes重新研究、insights研究结论。提高预算找舒适航班只选transport和budget，不能触发酒店或景点搜索。"
        "仅调整已选景点或用餐时刻只需days，不需要重新food研究。提到未修改部分或否定句不代表要重查。改城市顺序/停留时间/新增城市应选route。"
        "external仅指从出发地往返目的地的主交通，intercity指旅途中城市间的交通（包括境内航班）。只改某一段交通时，在target_transport_ids填写该段现有id，不影响其他段。用户问加入某地是否合理时，标为consideration，先比较绕路和时间成本，不能无解释删掉原城市。"
        "target_hotel_cities与research_cities只填确实要改的城市；不填表示整个受影响部分。"
        '返回 {"sections":[string],"target_hotel_cities":[string],"research_cities":[string],'
        '"transport_target":"external|intercity|all","target_transport_ids":[已有交通段id],"tasks":[string],"consideration":string,"reason":string}。',
        {
            "query": planner.query,
            "cities": previous.get("region", {}).get("cities", []),
            "days": [
                {
                    "day": d["day"],
                    "city": d["city"],
                    "places": [p["name"] for p in d["stops"]],
                    "activity_minutes": sum(p.get("duration_minutes", 90) for p in d["stops"]),
                    "transfer": d.get("type") == "transfer",
                }
                for d in previous.get("days", [])
            ],
            "hotels": [
                {"id": g.get("id"), "city": g["city"]} for g in previous.get("hotel_regions", [])
            ],
            "transport": [
                {
                    "id": t.get("id"),
                    "type": t["type"],
                    "scope": t.get("scope"),
                    "origin": t["origin"],
                    "destination": t["destination"],
                }
                for t in previous.get("transport", [])
            ],
        },
        "revision_scope",
    )
    targets = set(decision.get("target_transport_ids", []))
    matching = [t for t in previous.get("transport", []) if t.get("id") in targets]
    if targets and len(matching) == len(targets):
        scopes = {t.get("scope") for t in matching}
        decision["transport_target"] = next(iter(scopes)) if len(scopes) == 1 else "all"
    planner.change_request = decision
    scope = set(decision.get("sections", [])) & SECTIONS
    if before.get("destinations") != after.get("destinations"):
        scope.update(SECTIONS)
    if any(before.get(k) != after.get(k) for k in ("travel_dates", "duration_days")):
        scope.update(("route", "days", "hotel_regions", "transport", "food", "budget"))
    if before.get("traveler_count") != after.get("traveler_count"):
        scope.update(("hotel_regions", "transport", "budget"))
    if before.get("origin") != after.get("origin"):
        scope.update(("transport", "budget"))
    if "route" in scope:
        scope.update(("days", "hotel_regions", "transport", "food", "budget"))
    if "notes" in scope:
        scope.update(("days", "food", "insights", "budget"))
    if scope & {"days", "hotel_regions", "transport", "food"}:
        scope.add("budget")
    # A spending ceiling change alone never silently requests new hotels.
    if before.get("budget") != after.get("budget"):
        scope.add("budget")
    from travel_agent.reports.route import canonical_city

    planner.hotel_targets = [
        (canonical_city(previous["region"], name) or {}).get("name", name)
        for name in decision.get("target_hotel_cities", [])
    ]
    return scope, str(decision.get("reason") or "根据用户当前任务更新对应对象")


def preserve_failed_refresh(fresh, previous, label, issues, compatible=True):
    if fresh or not previous:
        return fresh
    issues.append(f"{label}本轮补查未成功；保留上一版候选及原检索时间。")
    retained = copy.deepcopy(previous)
    for item in retained:
        item["refresh_status"] = (
            "补查失败，保留历史候选" if compatible else "需求已变化，仅作历史参考，需重新报价"
        )
        item["compatible"] = compatible
    return retained
