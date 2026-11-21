from __future__ import annotations

from collections.abc import Iterable


def compact_messages(messages: Iterable[str], max_chars: int = 8000) -> str:
    """Create a deterministic fallback summary when no summarizer model is configured."""
    items = list(messages)
    if not items:
        return ""
    if sum(len(item) for item in items) <= max_chars:
        return "\n".join(items)
    recent = items[-6:]
    older = items[:-6]
    older_text = " ".join(older)
    return (
        "历史对话摘要：已压缩较早的对话；完整原始记录仍保存在事件日志。\n"
        f"较早对话概览：{older_text[:max_chars // 2]}\n"
        "最近对话：\n"
        + "\n".join(recent)
    )
