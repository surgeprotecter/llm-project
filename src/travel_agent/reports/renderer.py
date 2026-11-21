from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt

from travel_agent.models import TravelPlan
from travel_agent.persistence import Database

_WEATHER_LABELS = {
    "destination": "目的地",
    "city": "城市",
    "temperature": "气温",
    "weather": "天气",
    "condition": "天气",
    "rain_risk": "降雨风险",
    "advice": "出行建议",
    "forecasts": "天气预报",
    "date": "日期",
    "dayweather": "白天天气",
    "nightweather": "夜间天气",
    "daytemp": "白天气温",
    "nighttemp": "夜间气温",
}


def _weather_value(value) -> str:
    if isinstance(value, dict):
        return "；".join(
            f"{_WEATHER_LABELS.get(str(key), key)}：{_weather_value(item)}"
            for key, item in value.items()
            if item is not None and item != ""
        )
    if isinstance(value, list):
        return "；".join(_weather_value(item) for item in value if item is not None and item != "")
    return str(value)


def _weather_lines(weather: dict) -> list[str]:
    return [
        f"{_WEATHER_LABELS.get(str(key), key)}：{_weather_value(value)}"
        for key, value in weather.items()
        if value is not None and value != "" and value != [] and value != {}
    ]


class ReportRenderer:
    def __init__(self, database: Database, output_dir: str | Path):
        self.database = database
        self.output_dir = Path(output_dir)
        template_dir = Path(__file__).parent / "templates"
        self.environment = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape(["html"]))

    def render(self, plan: TravelPlan) -> tuple[Path, Path]:
        if not plan.report:
            plan.report = self._standard_context(plan)
        target_dir = self.output_dir / "reports" / plan.trip_id
        target_dir.mkdir(parents=True, exist_ok=True)
        version = self._version(plan)
        safe_destination = (plan.destination or "unknown").replace("/", "_").replace(" ", "_")
        markdown_path = target_dir / f"report_v{version}_{safe_destination}.md"
        html_path = target_dir / f"report_v{version}_{safe_destination}.html"
        if markdown := plan.report.get("chat_markdown"):
            template = self.environment.get_template("travel_itinerary_export.html")
            body = MarkdownIt("commonmark", {"html": False}).enable("table").render(markdown)
            content = template.render(title=plan.title, body=body, show_title="<h1>" not in body)
        else:
            markdown = self._markdown(plan)
            template = self.environment.get_template("travel_report_standard.html")
            content = template.render(plan=plan.model_dump(mode="python"), report=plan.report, weather_lines=_weather_lines(plan.weather))
        markdown_path.write_text(markdown, encoding="utf-8")
        html_path.write_text(content, encoding="utf-8")
        self.database.save_report(plan.trip_id, version, str(markdown_path), str(html_path), plan.model_dump(mode="json"))
        return markdown_path, html_path

    def _standard_context(self, plan: TravelPlan) -> dict:
        """Keep focused/older report payloads readable in the same layout."""
        from travel_agent.models import TripState
        trip = self.database.get_trip(plan.trip_id) or TripState(trip_id=plan.trip_id)
        days = []
        for day in plan.itinerary:
            stops = [{**a, "start_time": a.get("start_time", "建议时刻待定"),
                      "reason": a.get("reason", "候选地点，路线需进一步核验"),
                      "map_url": a.get("map_url") or a.get("source_url") or "https://www.amap.com/",
                      "source_url": a.get("source_url") or "https://www.amap.com/"} for a in day.activities]
            days.append({"day":day.day,"date":"","title":day.title,"city":plan.destination,"area":plan.destination,
                         "intro":"当前为专项查询结果，尚未形成完整路线。","stops":stops,"legs":[]})
        hotels = []
        for index, h in enumerate(plan.hotel_options[:2]):
            hotels.append({**h,"url":h.get("source_url") or h.get("url") or "https://www.booking.com/",
                           "baseline":index==0,"price_is_total":bool(h.get("total_price")),"total_price":h.get("total_price"),
                           "location":h.get("area") or h.get("location") or plan.destination,
                           "pros":["候选住宿，需核对具体位置"],"cons":["房型、税费和可订状态待确认"]})
        groups = [{"city":plan.destination,"area":plan.destination,"hotels":hotels}] if hotels else []
        return {"trip":trip.model_dump(mode="json"),"generated_at":datetime.now(UTC).isoformat(),"days":days,"point_count":sum(len(d["stops"]) for d in days),
                "hotel_regions":groups,"transport":[{"type":"交通候选","origin":trip.origin or "待补充","destination":plan.destination or "待补充","date":"","url":"https://www.12306.cn/index/" if any("train_no" in o for o in plan.transport_options) else "https://www.google.com/travel/flights","options":plan.transport_options,"raw_summary":""}] if plan.transport_options else [],"summary":plan.summary,"status":"专项查询 · 待完善",
                "budget":{"items":[{"category":"预算","label":"已有专项结果","basis":"需要补齐完整行程报价","amount":None,"status":"待测算"}],"finalized":False,"total":None,"reason":"当前专项查询未覆盖全部预算分项。"},
                "insights":[],"notes":plan.research_notes,"sources":plan.sources,"warnings":plan.warnings,
                "confirmations":["补齐日期和主要需求后，可继续生成完整旅行报告。"]}

    def _version(self, plan: TravelPlan) -> int:
        latest = self.database.latest_report(plan.trip_id)
        return int(latest["version"]) + 1 if latest else 1

    def _markdown(self, plan: TravelPlan) -> str:
        if plan.report.get("schema_version") == 2:
            return self.environment.get_template("travel_report_standard.md.j2").render(plan=plan, report=plan.report)
        lines = [f"# {plan.title}", "", plan.summary, ""]
        if plan.destination:
            lines.extend([f"**目的地：** {plan.destination}", ""])
        if plan.assumptions:
            lines.extend(["## 规划假设", "", *[f"- {item}" for item in plan.assumptions], ""])
        weather_lines = _weather_lines(plan.weather)
        if weather_lines:
            lines.extend(["## 天气参考", "", *[f"- {item}" for item in weather_lines], ""])
        lines.append("## 每日行程\n")
        for day in plan.itinerary:
            lines.extend([f"### 第 {day.day} 天：{day.title}", ""])
            for activity in day.activities:
                name = activity.get("name") or activity.get("title") or "未命名活动"
                reason = activity.get("reason") or activity.get("type") or ""
                url = activity.get("source_url") or activity.get("url") or activity.get("link")
                label = f"[{name}]({url})" if isinstance(url, str) and url.startswith(("http://", "https://")) else f"**{name}**"
                source_note = activity.get("source_note")
                source_text = f"（依据：{source_note}）" if source_note else ""
                lines.append(f"- {label}：{reason}{source_text}")
            if day.estimated_cost is not None:
                lines.append(f"- 当日预计花费：{day.estimated_cost:.0f} 元")
            lines.append("")
        if plan.hotel_options:
            lines.extend(["## 住宿候选", ""])
            for hotel in plan.hotel_options:
                url = hotel.get("source_url") or hotel.get("url") or hotel.get("propertyUrl")
                name = hotel.get("name") or hotel.get("title") or "住宿候选"
                label = f"[{name}]({url})" if isinstance(url, str) and url.startswith(("http://", "https://")) else str(name)
                price = hotel.get("nightly_price")
                total_price = hotel.get("total_price") or hotel.get("totalPrice")
                if hotel.get("price_is_total") is False:
                    price_text = f"，平台卡片展示价 {total_price} {hotel.get('currency', '')}（计价周期与币种待核对，不计入住宿总额）"
                else:
                    price_text = f"，约 {price} {hotel.get('currency', '元')}/晚" if price is not None else (f"，入住总价 {total_price} {hotel.get('currency', '')}" if total_price is not None else "，价格未提供")
                lines.append(f"- {label}：{hotel.get('area', hotel.get('location', ''))}{price_text}，评分 {hotel.get('rating', hotel.get('reviewScore', '未知'))}")
            lines.append("")
        if plan.research_notes:
            lines.extend(["## 攻略与口碑线索", ""])
            for note in plan.research_notes:
                title = note.get("title") or note.get("name") or "旅行笔记"
                url = note.get("url") or note.get("source_url") or note.get("link")
                label = f"[{title}]({url})" if isinstance(url, str) and url.startswith(("http://", "https://")) else title
                summary = note.get("content") or note.get("desc") or note.get("description") or ""
                lines.append(f"- {label}：{summary}")
            lines.append("")
        if plan.transport_options:
            lines.extend(["## 交通候选", ""])
            for option in plan.transport_options:
                route = option.get("route") or option.get("name") or option.get("flightNumber") or option.get("train_no") or "交通候选"
                duration = option.get("duration") or option.get("durationText") or ""
                price = option.get("estimated_price") or option.get("price") or option.get("price_amount")
                price_text = f"，价格 {price} {option.get('currency') or '元'}" if price is not None else "，价格未提供"
                url = option.get("source_url") or option.get("url") or option.get("bookingUrl")
                label = f"[{route}]({url})" if isinstance(url, str) and url.startswith(("http://", "https://")) else route
                lines.append(f"- {label}：{duration}{price_text}")
            lines.append("")
        lines.extend(["## 预算估算", ""])
        for key, value in plan.budget_breakdown.items():
            lines.append(f"- {key}：{value:.0f} 元")
        if plan.total_estimated_cost is not None:
            lines.extend([f"- **总计：{plan.total_estimated_cost:.0f} 元**", ""])
        else:
            lines.extend(["- **总计：暂无法可靠估算**", ""])
        if plan.warnings:
            lines.extend(["## 注意事项", "", *[f"- {item}" for item in plan.warnings], ""])
        if plan.sources:
            lines.extend(["## 信息来源", ""])
            for source in plan.sources:
                name = source.get("name", "外部来源")
                url = source.get("url")
                fetched_at = source.get("fetched_at", "")
                label = f"[{name}]({url})" if isinstance(url, str) and url.startswith(("http://", "https://")) else name
                lines.append(f"- {label}（检索时间：{fetched_at}）")
            lines.append("")
        return "\n".join(lines)
