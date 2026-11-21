from travel_agent.graph.agent_nodes import _real_plan_summary
from travel_agent.models import TravelPlan
from travel_agent.reports.renderer import ReportRenderer


def test_real_summary_preserves_provider_currency_and_never_invents_total() -> None:
    plan = TravelPlan(
        trip_id="t1",
        title="岘港机票查询",
        summary="真实检索结果",
        destination="岘港",
        transport_options=[
            {
                "route": "Hong Kong → Da Nang · Hong Kong Express UO558",
                "durationText": "16:50–17:45 · 115 分钟",
                "estimated_price": 146,
                "currency": "USD",
            }
        ],
    )

    summary = _real_plan_summary(plan, "report.md", "report.html")

    assert "146 USD" in summary
    assert "1019" not in summary
    assert "没有可靠的完整分项" in summary


def test_mock_report_renders_weather_in_markdown_and_html(database, tmp_path) -> None:
    plan = TravelPlan(
        trip_id="weather-test",
        title="成都行程规划",
        summary="Mock 测试行程",
        weather={"destination": "成都", "temperature": "18-27°C", "rain_risk": "中等", "advice": "携带雨伞"},
    )

    markdown_path, html_path = ReportRenderer(database, tmp_path).render(plan)

    assert "## 天气参考" in markdown_path.read_text(encoding="utf-8")
    assert "18-27°C" in html_path.read_text(encoding="utf-8")
