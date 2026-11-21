"""Audit and package real acceptance artifacts. No models, tools or synthetic data."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from travel_agent.models import TravelPlan
from travel_agent.persistence import Database
from travel_agent.reports import ReportRenderer
from travel_agent.reports.daily import sparse_tour_days

BASE = Path("output/adaptive-acceptance")
CASES = {
    "domestic-short": BASE / "domestic_short/20261116-191354",
    "international-short": BASE / "international_short/20261116-191958",
    "domestic-long": BASE / "domestic_long/reviewed-20261116-234657",
    "egypt-14": BASE / "international_long/20261116-191734",
}


def report(root, version=None):
    connection = sqlite3.connect(root / "session.sqlite3")
    row = connection.execute(
        "select version,payload_json,html_path,markdown_path from reports "
        + ("where version=?" if version else "")
        + " order by version desc limit 1",
        (version,) if version else (),
    ).fetchone()
    connection.close()
    return row, json.loads(row[1])["report"]


def main():
    target = BASE / "deliverables"
    target.mkdir(exist_ok=True)
    summary = {}
    for name, root in CASES.items():
        row, r = report(root)
        checks = {
            "quality": r["quality"]["passed"],
            "days": len(r["days"]) == r["trip"]["duration_days"],
            "substantive_days": not sparse_tour_days(r["days"], r["transport"]),
            "city_coverage": all(
                any(p.get("city_id") == c["id"] for p in r["places"]) for c in r["region"]["cities"]
            ),
            "hotel_coverage": all(g["hotels"] for g in r["hotel_regions"]),
            "budget_arithmetic": sum(i["low"] for i in r["budget"]["items"]) == r["budget"]["low"]
            and sum(i["high"] for i in r["budget"]["items"]) == r["budget"]["high"],
            "night_coverage": sum(g["nights"] for g in r["hotel_regions"])
            == r["trip"]["duration_days"]
            - 1
            - sum(bool(d.get("no_hotel")) for d in r["days"][:-1]),
        }
        if name == "international-short":
            _, before = report(root, 2)
            _, after = report(root, 3)
            checks.update(
                flight_revision_kept_hotels=before["hotel_regions"] == after["hotel_regions"],
                flight_revision_kept_notes=before["notes"] == after["notes"],
                flight_revision_kept_places=[s["id"] for d in before["days"] for s in d["stops"]]
                == [s["id"] for d in after["days"] for s in d["stops"]],
            )
        if name == "domestic-long":
            _, before = report(root, 1)
            _, after = report(root, 2)
            checks.update(
                hotel_revision_kept_days=before["days"] == after["days"],
                hotel_revision_kept_transport=before["transport"] == after["transport"],
                hotel_revision_kept_notes=before["notes"] == after["notes"],
                hotel_revision_kept_first_stay=before["hotel_regions"][0]
                == after["hotel_regions"][0],
            )
        if name == "egypt-14":
            checks["five_requested_cairo_places"] = len(
                r.get("mandatory_place_ids", [])
            ) >= 5 and all(
                i in {s["id"] for d in r["days"] for s in d["stops"]}
                for i in r["mandatory_place_ids"]
            )
            checks["flight_exit_city"] = r["days"][-1]["city"] == r["region"]["cities"][0]["name"]
        summary[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "source_root": str(root),
            "version": row[0],
            "html": str((target / f"{name}.html").resolve()),
            "budget": {k: r["budget"][k] for k in ["low", "high", "currency"]},
            "quality_issues": r["quality"]["issues"],
        }
        database = Database(root / "session.sqlite3")
        renderer = ReportRenderer(database, root)
        payload = json.loads(row[1])
        plan = TravelPlan.model_validate(payload)
        (target / f"{name}.html").write_text(
            renderer.environment.get_template("travel_report_standard.html").render(
                plan=plan.model_dump(mode="python"), report=r, weather_lines=[]
            )
        )
        (target / f"{name}.md").write_text(renderer._markdown(plan))
        database.close()
        (target / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))
    records = []
    for path in BASE.rglob("*model-trace.json"):
        if "deliverables" in path.parts:
            continue
        try:
            rows = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        records.extend(
            {"file": str(path), **r} for r in rows if r.get("status") != "local_cache_hit"
        )
    usage = {
        "api_attempts": len(records),
        "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in records),
        "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in records),
        "attempts_without_usage": sum(not r.get("usage") for r in records),
        "includes_failed_debug_runs": True,
        "note": "包括本次任务全部保留的成功、失败与重试记录；超时未返回usage的请求未推算token，不代表每次旅行请求固定消耗。",
    }
    (target / "acceptance.json").write_text(
        json.dumps({"cases": summary, "usage": usage}, ensure_ascii=False, indent=2)
    )
    lines = [
        "# 真实对话验收与报告",
        "",
        "测试输入不指定工具，覆盖模糊追问、国内外、短途与12/14天、多城市及局部修改。所有验收使用真实模型和真实工具；失败记录保留。",
        "",
        "| 场景 | 最终版本 | 检查 | 报告 |",
        "| --- | ---: | --- | --- |",
    ]
    for name, item in summary.items():
        lines.append(
            f"| {name} | {item['version']} | {'通过' if item['passed'] else '未通过'} | [{name}.html]({name}.html) |"
        )
    lines += [
        "",
        "## 用量",
        "",
        f"本次全部保留调试记录：{usage['api_attempts']}次API尝试，已返回usage合计输入{usage['input_tokens']}、输出{usage['output_tokens']} token；{usage['attempts_without_usage']}次无usage。含失败及重试，不是单次规划费用。",
        "",
        "局部修改实测：苏杭只换杭州酒店为2次模型调用、2081输入+371输出token；岘港只改舒适航班为2次调用、2012输入+427输出token。两者只触发相应类别的真实工具。",
        "",
        "酒店/交通候选与预算区间未锁价。运营商参考、道路排程估算、规划预留分别标明；海上或全天体验按整段活动安排，不伪造道路可达性。",
        "",
        "## 证据目录",
        "",
    ]
    lines += [
        f"- {name}: `{item['source_root']}`，含每轮报告、模型与工具trace及followup验收记录。"
        for name, item in summary.items()
    ]
    all_attempts = []
    for trace_file in BASE.rglob("*model-trace.json"):
        if "deliverables" in trace_file.parts:
            continue
        try:
            attempts = json.loads(trace_file.read_text())
            if isinstance(attempts, list):
                all_attempts.extend(x for x in attempts if x.get("status") != "local_cache_hit")
        except (ValueError, OSError):
            continue
    usage = {
        "api_attempts": len(all_attempts),
        "input_tokens": sum(x.get("usage", {}).get("input_tokens", 0) for x in all_attempts),
        "output_tokens": sum(x.get("usage", {}).get("output_tokens", 0) for x in all_attempts),
        "without_usage": sum(not x.get("usage") for x in all_attempts),
    }
    (target / "model-usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2))
    lines += [
        "",
        "## 全部调试用量",
        "",
        f"累计真实API尝试{usage['api_attempts']}次，输入{usage['input_tokens']}、输出{usage['output_tokens']} token；包含失败、重试和早期调试，不代表一次规划的用量。费用以服务商账单为准。",
    ]
    (target / "acceptance.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "cases": {
                    k: {
                        "passed": v["passed"],
                        "failed": [c for c, p in v["checks"].items() if not p],
                    }
                    for k, v in summary.items()
                },
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not all(item["passed"] for item in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
