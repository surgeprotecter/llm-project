"""Project each report stage onto only the evidence it needs."""

from __future__ import annotations


def note_view(notes, focus, limit=12):
    # The extraction model reads the actual bodies. Do not select paragraphs by keyword.
    return [{"id": n["id"], "title": n.get("title", ""), "content": n["content"]}
            for n in notes[:limit]]


def model_view(stage, value):
    value = dict(value)
    if stage in {"extract_places", "food_extract"}:
        value["notes"] = note_view(
            value.get("notes", []),
            "food" if stage == "food_extract" else "places",
            limit=8 if stage == "food_extract" else 12,
        )
    if stage == "match_places":
        value["candidates"] = [
            {
                "place": {
                    k: b["place"].get(k) for k in ("name", "city", "kind", "query_name", "aliases")
                },
                "candidates": [
                    {
                        k: c.get(k)
                        for k in ("id", "name", "city", "district", "osm_value", "type", "location")
                    }
                    for c in b["candidates"]
                ],
            }
            for b in value.get("candidates", [])
        ]
    if stage == "schedule":
        value["places"] = [
            {k: p.get(k) for k in ("id", "name", "city", "area", "location")}
            for p in value["places"]
        ]
        transport = []
        for t in value.get("transport", []):
            options = t.get("options", [])
            chosen = next(
                (o for o in options if o.get("baseline")),
                next((o for o in options if o.get("seat_available")), next(iter(options), {})),
            )
            transport.append(
                {
                    "type": t["type"],
                    "origin": t["origin"],
                    "destination": t["destination"],
                    "date": t["date"],
                    "option": {
                        k: chosen.get(k)
                        for k in ("start_time", "arrive_time", "departure_time")
                        if chosen.get(k)
                    },
                    "directions": [
                        {k: d.get(k) for k in ("direction", "departure_time", "arrival_time")}
                        for d in chosen.get("directions", [])
                    ],
                }
            )
        value["transport"] = transport
        if value.get("previous_failed_schedule"):
            value["previous_failed_schedule"] = [
                {
                    "day": d["day"],
                    "city": d["city"],
                    "stops": [
                        {k: s.get(k) for k in ("id", "start_time", "duration_minutes")}
                        for s in d["stops"]
                    ],
                    "legs": [{k: leg.get(k) for k in ("minutes", "verified")} for leg in d["legs"]],
                    "warning": d.get("timing_warning"),
                    "available_until": d.get("available_until"),
                }
                for d in value["previous_failed_schedule"]
            ]
    return value


def stage_requirements(stage, state):
    keys = {
        "origin",
        "destinations",
        "travel_dates",
        "duration_days",
        "traveler_count",
        "budget",
        "preferences",
        "hard_constraints",
        "confirmed_decisions",
    }
    if stage in {"region", "airports"}:
        keys = {"origin", "destinations", "duration_days", "hard_constraints"}
    elif stage in {"match_places", "note_selection", "supplement_notes", "cost_evidence"}:
        keys = {"destinations"}
    return {k: state[k] for k in sorted(keys)}
