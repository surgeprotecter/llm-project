"""Keep approved route requirements separate from a model's current step."""
from __future__ import annotations

import unicodedata

from travel_agent.persistence.runtime import canonical


def approved_plan_view(state):
    approval = state.get("plan_approval") or {}
    if state.get("request_scope", {}).get("scenario") != "execute" or approval.get("status") != "approved":
        return {}
    if not approval.get("proposal"):
        return {}
    return {
        "version": approval["version"],
        "proposal": approval["proposal"],
        "user_confirmation": approval.get("user_confirmation", ""),
    }


def extract_route_contract(gateway, approved, requirements):
    result = gateway.structured_extract(
        "从已批准的旅行提案提取路线约束，不生成新路线。提案正文是已批准版本；确认原话只补充明确回答，"
        "答复中泛指'可以'不改变任何规划。另列出的用户要求可帮助复原范围。"
        "遍历正文的行程表和路线摘要，保留每一个实际安排停留或游玩的城市/区域，包括合写在同一行的多个城市。"
        "仅有经过/换乘、仅举例的城市或明确标成可选且未决定的备选地不算承诺，不把国家、机场和景点当成额外停留地。"
        "示例日期待查票不代表表里已安排游览的城市可删除。name用原文中的准确地名；aliases可用常用译名或英文。"
        "具体约定住宿/停留天数写上下限；如果只规定了区域合计或只是建议，不要把它当成每个城市的固定天数。"
        "明确只留X天则上下限都为X；X-Y天保留范围；尚未确定用null。total_days仅提取明确的固定全程天数，不为范围定值。"
        "reserved只提取提案单独列出的、没有算进任何目的地停留的日子。outbound=出发、return=返程、flex=机动或一并写作机动与返程的合计。"
        "同一天不能记两次；城市段内已包含的抵离时间不再计入。没有单独分配的类型填0。"
        '只返回JSON {"total_days":integer或null,"reserved":{"outbound":integer,"return":integer,"flex":integer},'
        '"stops":[{"name":string,"aliases":[string],"min_days":integer或null,"max_days":integer或null}]}。',
        canonical({"approved_plan": approved, "requirements": requirements}),
        stage="approved_route_contract", output_tokens=1600,
    )
    if not isinstance(result, dict) or not isinstance(result.get("stops"), list):
        raise ValueError("未能从已批准提案整理路线约束")
    # Use grounded stop names. Reject malformed bounds instead of silently loosening them.
    source = approved["proposal"] + "\n" + approved["user_confirmation"] + "\n" + "\n".join(requirements)
    stops = []
    seen = set()
    for item in result["stops"]:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.strip() or name not in source:
            raise ValueError("路线约束出现了提案里没有的停留地点")
        low, high = item.get("min_days"), item.get("max_days")
        if any(v is not None and (type(v) is not int or v < 1) for v in (low, high)) or (
            low is not None and high is not None and high < low
        ):
            raise ValueError(f"{name}的停留天数约束无效")
        if key(name) in seen:
            raise ValueError(f"提案中的{name}应按整个行程合并计算停留天数")
        seen.add(key(name))
        aliases = item.get("aliases")
        stops.append({"name": name, "aliases": [a for a in aliases if isinstance(a, str) and key(a)]
                      if isinstance(aliases, list) else [], "min_days": low, "max_days": high})
    total = result.get("total_days")
    if total is not None and (type(total) is not int or total < 1):
        raise ValueError("提案的总天数约束无效")
    raw = result.get("reserved")
    raw = raw if isinstance(raw, dict) else {}
    reserved = {k: raw.get(k, 0) for k in ("outbound", "return", "flex")}
    if any(type(v) is not int or v < 0 for v in reserved.values()) or (
        total is not None and sum(reserved.values()) >= total
    ):
        raise ValueError("提案中的独立交通与机动天数无效")
    return {"schema": 2, "version": approved["version"], "total_days": total, "stops": stops, "reserved": reserved}


def key(name):
    value = unicodedata.normalize("NFKC", name).casefold().strip().removesuffix("市")
    return "".join(c for c in value if c.isalnum())


def city_keys(city):
    aliases = city.get("aliases")
    return {key(n) for n in [city.get("name"), city.get("city"), city.get("query"),
                             *(aliases if isinstance(aliases, list) else [])]
            if isinstance(n, str) and key(n)}


def reserve_kinds(city):
    return [*city.get("reserve_before", []), *city.get("reserve_after", [])]


def fit_visit_counts(contract, cities, target):
    """Repair arithmetic without changing the model's ordering or adding stops."""
    if not contract or not contract["stops"] or not cities:
        return
    aliases = [{key(s["name"]), *(key(a) for a in s["aliases"])} for s in contract["stops"]]
    matches = [[i for i, names in enumerate(aliases) if names & city_keys(c)] for c in cities]
    bound_stops = [i for row in matches for i in row]
    if any(len(row) > 1 for row in matches) or sorted(bound_stops) != list(range(len(aliases))):
        return  # Missing or split stops need a semantic route repair.
    bounds = []
    for row in matches:
        stop = contract["stops"][row[0]] if row else {}
        bounds.append((stop.get("min_days") or 1, stop.get("max_days") or target))
    if not sum(lo for lo, _ in bounds) <= target <= sum(hi for _, hi in bounds):
        return
    counts = [min(hi, max(lo, c["days"])) for c, (lo, hi) in zip(cities, bounds)]
    remaining = target - sum(counts)
    for i, (lo, hi) in enumerate(bounds):
        amount = min(remaining, hi - counts[i]) if remaining > 0 else max(remaining, lo - counts[i])
        counts[i] += amount
        remaining -= amount
    for city, days in zip(cities, counts):
        city["days"] = days


def route_issues(contract, cities):
    if not contract:
        return []
    counts = [c.get("days") if type(c.get("days")) is int else 0 for c in cities]
    total = contract.get("total_days")
    issues = []
    if total is not None and sum(counts) != total:
        issues.append(f"全程批准的是{total}天，目前排了{sum(counts)}天")
    visit_counts = [n - len(reserve_kinds(city)) for n, city in zip(counts, cities)]
    reserved = [kind for city in cities for kind in reserve_kinds(city)]
    prefix = {"outbound": "出发", "return": "返程", "flex": "机动或返程"}
    for kind, expected in contract.get("reserved", {}).items():
        if reserved.count(kind) != expected:
            issues.append(f"应单列{expected}天{prefix[kind]}时间，目前单列{reserved.count(kind)}天")
    names = [city_keys(city) for city in cities]
    for stop in contract["stops"]:
        aliases = {key(stop["name"]), *(key(a) for a in stop["aliases"])}
        matched = [n for n, city in zip(visit_counts, names) if city & aliases and n > 0]
        name = stop["name"]
        if not matched:
            issues.append(f"漏排了已批准的{name}")
            continue
        days = sum(matched)
        low, high = stop["min_days"], stop["max_days"]
        if (low is not None and days < low) or (high is not None and days > high):
            expected = f"{low}天" if low == high else f"{low}–{high}天" if low and high else (
                f"至少{low}天" if low else f"最多{high}天"
            )
            issues.append(f"{name}批准的是{expected}，目前排了{days}天")
    return issues


class ApprovedRouteConflict(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("；".join(issues))
