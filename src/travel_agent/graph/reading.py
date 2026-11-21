"""Executable page cursors and durable coverage of saved tool data."""
from __future__ import annotations

from travel_agent.context.agent_context import tool_view


def next_args(result_id, field, offset):
    return {"result_id": result_id, "field": field, "offset": offset}


def remove_preview_markers(value):
    if isinstance(value, list):
        return [remove_preview_markers(item) for item in value
                if not (isinstance(item, dict) and set(item) == {"remaining_items"})]
    if isinstance(value, dict):
        return {key: remove_preview_markers(item) for key, item in value.items()}
    return value


def preview_result(root, result_id):
    view = remove_preview_markers(tool_view(root, limit=3))
    pages = []
    for field, original in root.items():
        shown = view.get(field)
        if isinstance(original, list) and isinstance(shown, list) and shown:
            pages.append(page_info(result_id, field, 0, len(shown), len(original)))
    pending = [page for page in pages if page["next_args"]]
    response = {"data": view, "available_fields": list(root), "pages": pages}
    if len(pending) == 1:
        response.update(next_offset=pending[0]["next_offset"], next_args=pending[0]["next_args"])
    return response


def page_info(result_id, field, offset, returned, total):
    stop = offset + returned
    more = stop if returned and stop < total else None
    return {"field": field, "offset": offset, "returned": returned, "total": total,
            "next_offset": more, "next_args": next_args(result_id, field, more) if more is not None else None}


def first_unread(ranges):
    return ranges[0][1] if ranges and ranges[0][0] == 0 else 0


def page_target(progress, field):
    """Resolve continuation without relying on offsets copied by the model."""
    if field:
        return field, progress.get(field)
    if len(progress) == 1:
        return next(iter(progress.items()))
    unread = [(name, entry) for name, entry in progress.items() if entry.get("next_args")]
    if len(unread) == 1:
        return unread[0]
    return "", None


def exhausted_page(result_id, field, total):
    return {"success": True, "prepared_view": True, "unchanged": True, "exhausted": True,
            "result_id": result_id, "field": field, "total": total,
            "next": "此字段已读完。用已取得的候选完成当前比较或保存结论。"}


def cover(ranges, start, stop):
    updated = []
    for left, right in sorted([*ranges, [start, stop]]):
        if updated and left <= updated[-1][1]:
            updated[-1][1] = max(updated[-1][1], right)
        else:
            updated.append([left, right])
    return updated


def remember_pages(progress, result):
    """Track actual returned ranges, independent of the entry point or limit."""
    pages = result.get("pages") or ([result] if "returned" in result and "field" in result else [])
    pages = [page for page in pages if page["returned"]]
    result_id = result.get("result_id")
    if not result_id or not pages:
        return result
    entry = progress.pop(result_id, {})
    progress[result_id] = entry
    new = False
    continuations = []
    for page in pages:
        field = page["field"]
        start = page["offset"]
        stop = start + page["returned"]
        old = entry.get(field, {}).get("ranges", [])
        ranges = cover(old, start, stop)
        new |= old != ranges
        cursor = first_unread(ranges)
        continuation = next_args(result_id, field, cursor) if cursor < page["total"] else None
        entry[field] = {"ranges": ranges, "total": page["total"], "next_args": continuation}
        if continuation:
            continuations.append(continuation)
    if len(continuations) == 1:
        result["next_args"] = continuations[0]
        result["next_offset"] = continuations[0]["offset"]
    elif not continuations:
        result.update(next_args=None, next_offset=None, exhausted=True)
    if new:
        return result
    result = {"success": True, "unchanged": True, "prepared_view": True, "result_id": result_id,
              "next": "这部分已读。需要新记录时用未读参数，需要某条细节时指定完整字段路径。"}
    if len(continuations) == 1:
        result["next_args"] = continuations[0]
    elif continuations:
        result["next_options"] = continuations
    return result
