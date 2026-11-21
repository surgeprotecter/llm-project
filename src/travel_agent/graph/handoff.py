"""Carry completed steps and their compact evidence into final synthesis."""
from __future__ import annotations

from travel_agent.context.agent_context import project


def select_fields(data, fields):
    return {key: data[key] for key in fields if key in data and data[key] is not None}


def saved_material(store, state):
    selected = {}
    for choice in state.get("selected_results", []):
        selected.setdefault(choice["result_id"], []).append(choice["item_index"])
    outputs = []
    for task in state.get("task_plan", []):
        if task["status"] != "completed":
            continue
        cards = []
        for ref in dict.fromkeys(task.get("result_refs", [])):
            if ref.startswith(("workspace:", "artifact:")):
                continue
            record = store.result(state["thread_id"], ref)
            if record and record["result"].get("success"):
                cards.append(material_card(record, ref, selected.get(ref, []), bool(task.get("summary"))))
        outputs.append({"task": task["description"], "summary": task.get("summary"), "evidence": cards})
    return outputs


def material_card(record, ref, selected, summarized):
    """Preserve price units and dates without copying provider payloads."""
    result = record["result"]
    data = result.get("data", {})
    card = {"ref": ref, "tool": record["tool_name"], "fetched_at": result.get("fetched_at"),
            "query": select_fields(record.get("arguments", {}), (
                "origin", "destination", "departure_date", "return_date", "checkIn", "checkOut",
                "adults", "rooms", "passengers", "query", "keyword",
            ))}
    limit = 1 if summarized else 3
    if isinstance(properties := data.get("properties"), list):
        indices = list(dict.fromkeys([*selected, *range(len(properties))]))[:limit]
        card["hotels"] = [{"selected": i in selected, **select_fields(properties[i], (
            "name", "location", "reviewScore", "reviewCount", "currency", "priceIsTotal",
            "totalPrice", "pricePerNight", "priceNote", "url",
        ))} for i in indices if i < len(properties)]
        for item in card["hotels"]:
            if "url" in item:
                item["url"] = str(item["url"]).split("?", 1)[0]
    elif isinstance(flights := data.get("flights"), list):
        indices = list(dict.fromkeys([*selected, *range(len(flights))]))[:limit]
        card["quote_context"] = data.get("quote_context")
        card["flight_count"] = len(flights)
        card["flights"] = [select_fields(flights[i], ("primary_airline_name", "price", "currency", "booking_url"))
                           for i in indices if i < len(flights)]
    elif isinstance(dates := data.get("dates"), list):
        card["date_options"] = [select_fields(item, ("date", "return_date", "currency", "price"))
                                for item in dates[:limit]]
    elif isinstance(results := data.get("results"), list):
        card["sources"] = [{**select_fields(item, ("title", "url")),
                            "excerpt": str(item.get("body") or item.get("description") or "")[:280]}
                           for item in results[:min(limit, 2)]]
    else:
        card["preview"] = project(data, limit=2, text_limit=300)
    return card
