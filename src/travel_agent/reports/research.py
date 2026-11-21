"""Research the missing city/topic, preserving body citations and compact inputs."""

from __future__ import annotations

from travel_agent.reports.route import city_notes


def research_city(planner, city, notes, old_points, purpose="all"):
    progress = getattr(planner, "research_progress", {}).setdefault(city["id"], {})
    checkpoint = getattr(planner, "checkpoint", lambda: None)
    if progress.get("purpose") != purpose:
        progress.pop("extracted", None)
    progress["purpose"] = purpose
    if purpose in {"food", "repair", "coverage_repair"} and progress.get("completed"):
        progress.clear()
        progress["purpose"] = purpose
    progress.update(phase="reading_notes", completed=False)
    checkpoint()
    if purpose == "coverage_repair":
        expansion = planner.ask(
            '根据偏好补查当前城市的具体半日/一日体验，排除仅购物补给。返回两个包含城市名的简短关键词 {"keywords":[string]}。',
            {"city": city["name"], "preferences": planner.trip.preferences},
            "keywords",
        )
        notes = planner.extra_notes(
            expansion.get("keywords", [f"{city['name']}一日游体验"]), notes, limit=3
        )
    related = city_notes(notes, city)
    needs_body = len(related) < 2
    if needs_body:
        keywords = list(
            dict.fromkeys(
                [
                    f"{city['name']}旅行攻略",
                    *city.get("search_keywords", []),
                    f"{city['name']}咖啡餐厅小众店",
                ]
            )
        )[:2]
        notes = planner.extra_notes(keywords, notes, limit=3)
        related = city_notes(notes, city)
    if not related:
        planner.issues.append(f"{city['name']}未读取到相关旅行笔记正文，需要补查。")
        return notes, old_points, []
    limit = min(18, max(5, int(city["days"]) * 3))
    data = progress.get("extracted") or planner.ask(
        "只为指定城市从真实正文提取独立可安排的体验，包括景点、餐馆小吃咖啡、商店；活动仅在符合用户偏好时选。"
        "mention必须逐字出现在引用正文，name可以是核验用的规范名称，但不能改变实体。不造店名、票价或营业时间。远郊一日游可归到住宿基地城市，但记录实际地名。"
        "kind取sightseeing/meal/cafe/shop/activity。duration_minutes是规划建议时长，不是核实营业信息。"
        "超市/普通补给购物不能作为主要游览体验，除非用户明确要求。避免只摘著名景点，兼顾用户偏好与正文丰富度。给足多日不同体验，不能多个景点拼成一个名字。"
        "query_name/aliases使用通行英文或当地名供地图检索；city固定为给出的基地中文名。"
        "活动如无明确地点可返回location_query及source_quote作为经验依据，不造经营者和预订链接。"
        '返回 {"places":[{"name":string,"mention":string,"kind":string,"city":string,"query_name":string,"aliases":[string],'
        '"duration_minutes":number,"reason":string,"note_ids":[string],"recommended_items":[string],'
        '"price_reference":string}],"insights":[{"title":string,"text":string,"note_ids":[string]}]}。',
        {
            "city": city["name"],
            "days": city["days"],
            "candidate_limit": limit,
            "query": planner.query,
            "notes": related[-5:],
            "purpose": purpose,
        },
        "extract_places",
    )
    progress.update(extracted=data, phase="resolving_places")
    checkpoint()
    raw = []
    for point in data.get("places", [])[:limit]:
        point = {**point, "city": city["name"], "city_id": city["id"]}
        point["kind"] = (
            point.get("kind")
            if point.get("kind") in {"sightseeing", "meal", "cafe", "shop", "activity"}
            else "sightseeing"
        )
        source_text = "\n".join(
            n["content"] for n in related if n["id"] in point.get("note_ids", [])
        )
        reference = str(point.get("price_reference") or "")
        import re

        if reference and (
            not any(char.isdigit() for char in reference)
            and reference not in source_text
            or any(num not in source_text for num in re.findall(r"\d+(?:\.\d+)?", reference))
        ):
            point["price_reference"] = "未取得正文明确价格，预算按规划预留。"
        point["recommended_items"] = [
            v for v in point.get("recommended_items", []) if v in source_text
        ]
        raw.append(point)
    existing_names = {p["name"] for p in old_points}
    points = planner.resolve([p for p in raw if p["name"] not in existing_names], related)
    # No unverified location is silently scheduled. Keep findings and gaps visible.
    unique = {p["id"]: p for p in [*old_points, *points]}
    ids = {n["id"] for n in related}
    insights = [
        i for i in data.get("insights", []) if i.get("note_ids") and set(i["note_ids"]) <= ids
    ]
    primary = [
        p for p in unique.values() if p.get("kind", "sightseeing") in {"sightseeing", "activity"}
    ]
    target = max(2, int(city["days"]) * 2 - 1)
    if len(primary) < target and purpose not in {"repair", "coverage_repair"}:
        extra = planner.extra_notes([f"{city['name']}小众景点一日游"], notes, limit=2)
        if len(extra) > len(notes):
            return research_city(planner, city, extra, list(unique.values()), purpose="repair")
    if len(primary) < 2:
        planner.issues.append(f"{city['name']}可定位体验不足，尚不能形成完整游览安排。")
    return notes, list(unique.values()), insights


def research_requested(planner, missing, notes):
    """Resolve explicitly requested places even if general travel notes omitted them."""
    import hashlib

    from travel_agent.reports.route import canonical_city

    requests = []
    pages = []
    user_text = " ".join(
        [
            planner.query,
            *planner.trip.hard_constraints,
            planner.previous.get("revision", {}).get("query", ""),
        ]
    )
    for item in missing[:4]:
        name = str(item.get("name") or "")
        city = canonical_city(planner.geo, item.get("city"))
        if not name or name not in user_text or not city:
            continue
        requests.append({**item, "city": city["name"], "city_id": city["id"]})
        data = planner.call(
            "web_search",
            {
                "query": f"{item.get('query_name') or name} {city['query']} official visitor information",
                "limit": 2,
            },
        )
        for page in data.get("results", []):
            content = str(page.get("body") or page.get("description") or "")
            if not content or not page.get("url"):
                continue
            pages.append(
                {
                    "id": "web:" + hashlib.sha256(page["url"].encode()).hexdigest()[:16],
                    "title": page.get("title", "网页检索摘录"),
                    "content": content,
                    "url": page["url"],
                    "source": "网页检索摘录",
                }
            )
    if not requests or not pages:
        return notes, [], [f"指定地点尚未取得可用资料：{r['name']}" for r in requests]
    data = planner.ask(
        "只处理用户明确指定但一般攻略遗漏的地点。从真实网页摘录确认对应实体，mention必须逐字出现在某条摘录。"
        "name保留用户指定名称；query_name/aliases提取通行英文或当地名称供地图检索；不造实体、票价或营业信息。"
        "duration_minutes是规划建议时长。只能给与requests一致的地点与city，无法确认就省略。"
        '返回 {"places":[{"name":string,"city":string,"mention":string,"kind":"sightseeing|activity|shop|cafe|meal",'
        '"query_name":string,"aliases":[string],"duration_minutes":number,"reason":string,"note_ids":[string]}]}。',
        {"requests": requests, "notes": [{**p, "content": p["content"][:3200]} for p in pages]},
        "requested_places",
    )
    raw = []
    for point in data.get("places", []):
        req = next((r for r in requests if r["name"] == point.get("name")), None)
        if req:
            raw.append({**point, "city": req["city"], "city_id": req["city_id"]})
    found = planner.resolve(raw, pages)
    gaps = [
        f"指定地点尚未核验：{r['name']}"
        for r in requests
        if not any(p["name"] == r["name"] for p in found)
    ]
    return list({n["id"]: n for n in [*notes, *pages]}.values()), found, gaps
