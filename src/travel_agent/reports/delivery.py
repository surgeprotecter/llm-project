"""Render a delivery receipt from validated artifact facts, not model arithmetic."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


def completeness_issues(report):
    """Verify the standard report's content independently of its success flag."""
    if report.get("schema_version") != 2 or report.get("chat_markdown"):
        return ["需要标准旅行报告的数据。"]
    days, trip = report.get("days", []), report.get("trip", {})
    start = (trip.get("travel_dates") or {}).get("start_date")
    date_order = [d.get("date") for d in days]
    expected_dates = []
    if isinstance(start, str) and type(trip.get("duration_days")) is int:
        try:
            first = date.fromisoformat(start)
            expected_dates = [(first + timedelta(days=i)).isoformat() for i in range(trip["duration_days"])]
        except (ValueError, OverflowError):
            pass
    issues = []
    if not days or date_order != expected_dates or [d.get("day") for d in days] != list(range(1, len(days) + 1)):
        issues.append("逐日行程的日期或天数不完整。")
    cities = (report.get("region") or {}).get("cities") or []
    if not cities or not report.get("places") or not report.get("sources"):
        issues.append("缺少覆盖行程的地点或真实检索来源。")
    if not any(n.get("content") and n.get("url") for n in report.get("notes", [])):
        issues.append("缺少有来源链接的旅行笔记或检索正文。")
    if any(not (d.get("stops") or d.get("type") == "transfer"
                or d.get("type") == "rest" and d.get("approved_reserve_kind") == "flex") for d in days):
        issues.append("至少一个游览日缺少具体安排。")
    if trip.get("origin") and (report.get("region") or {}).get("entry_mode") != "none" and not any(
        t.get("options") for t in report.get("transport", []) if t.get("scope") == "external"
    ):
        issues.append("缺少有依据的往返交通。")
    stays = report.get("hotel_regions", [])
    nights = sum(not day.get("no_hotel") for day in days[:-1])
    if sum(g.get("nights") or 0 for g in stays) != nights or any(
        not g.get("hotels") or g.get("compatible") is False for g in stays
    ):
        issues.append("住宿候选未覆盖实际需要的夜晚。")
    budget = report.get("budget") or {}
    items = budget.get("items") or []
    if not items or not budget.get("currency") or any(
        not isinstance(budget.get(bound), int | float)
        or any(not isinstance(item.get(bound), int | float) for item in items)
        or abs(sum(item[bound] for item in items) - budget[bound]) > .01
        for bound in ("low", "high")
    ):
        issues.append("缺少完整的分项预算或分项合计不一致。")
    return issues


def standard_report_ready(report):
    return bool(report.get("quality", {}).get("passed")) and not completeness_issues(report)


def delivered_report(row, base_version, *, allow_chat=False):
    if not row or row["version"] <= base_version:
        return False
    report = json.loads(row["payload_json"]).get("report") or {}
    chat = bool(report.get("chat_markdown"))
    if (chat and not allow_chat) or (not chat and not standard_report_ready(report)):
        return False
    try:
        html = Path(row["html_path"]).read_text(encoding="utf-8").strip().lower()
        if not (html.startswith("<!doctype html>") and html.endswith("</html>")):
            return False
        if chat:
            return Path(row["markdown_path"]).read_text(encoding="utf-8") == report["chat_markdown"]
        return Path(row["markdown_path"]).stat().st_size > 0 and all(
            f'id="{section}"' in html for section in ("overview", "days", "stay", "cost", "research", "sources")
        )
    except (OSError, UnicodeError):
        return False


def receipt(row, note=None):
    report = json.loads(row["payload_json"])["report"]
    if report.get("chat_markdown"):
        return "\n\n".join(["已原样导出聊天行程。",
                              f"[聊天行程 HTML]({Path(row['html_path']).resolve()})",
                              f"[Markdown]({Path(row['markdown_path']).resolve()})"])
    links = [f"[HTML 攻略]({Path(row['html_path']).resolve()})", f"[Markdown]({Path(row['markdown_path']).resolve()})"]
    people = report["trip"].get("traveler_count") or 1
    lines = [f"已生成第 {row['version']} 版攻略，包含 {len(report['days'])} 天行程。"]
    if note and len(note) < 500 and not re.search(r"\d|[¥￥$]", note):
        lines.append(note)
    for transfer in report.get("transport", []):
        option = next((o for o in transfer.get("options", []) if o.get("baseline")), {})
        if option.get("directions") and option.get("per_person_price") is not None:
            lines.append(f"航班：每人往返参考 {option['per_person_price']:,.2f} {option['currency']}；"
                         f"{people} 人合计参考 {option.get('group_estimated_price', option['per_person_price'] * people):,.2f} {option['currency']}。")
    stays = report.get("hotel_regions", [])
    totals = defaultdict(float)
    missing_quotes = 0
    for stay in stays:
        hotel = next((h for h in stay.get("hotels", []) if h.get("baseline")), {})
        if hotel.get("price_is_total") and hotel.get("total_price") is not None:
            totals[hotel["currency"]] += hotel["total_price"]
        else:
            missing_quotes += 1
    text = "、".join(f"{amount:,.2f} {currency}" for currency, amount in totals.items())
    lines.append(f"住宿：{len(stays)} 段，共 {sum(g.get('nights') or 0 for g in stays)} 晚。"
                 + (f"所选基准房费合计 {text}，税费范围见报告。" if text else "部分报价尚待核实。"))
    if missing_quotes:
        lines.append(f"上述房费仅合计已取得明确报价的部分，另有 {missing_quotes} 段在预算中使用预留。")
    budget = report.get("budget", {})
    if budget.get("low") is not None:
        lines.append(f"全程估算：全体 {budget['low']:,}–{budget['high']:,} {budget['currency']}；"
                     f"人均 {budget['per_person_low']:,}–{budget['per_person_high']:,} {budget['currency']}。")
        preference = report["trip"].get("budget") or {}
        cap = preference.get("max_amount")
        if cap is None:
            cap = budget.get("target")
        if cap is not None and budget["high"] > cap:
            explanation = "预计超出" if budget["low"] > cap else "估算上限可能超出"
            lines.append(f"{explanation}预算 {cap:,.0f} {budget['currency']}。")
    lines.extend(links)
    return "\n\n".join(lines)
