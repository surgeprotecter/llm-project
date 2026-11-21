from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from travel_agent.models import ToolResult, ToolSpec
from travel_agent.tools.base import TravelTool


def _result(name: str, data: dict[str, Any], summary: str) -> ToolResult:
    return ToolResult(
        success=True,
        tool_name=name,
        data=data,
        summary=summary,
        source_type="mock",
        source_name=f"mock_{name}",
        valid_until=datetime.now(UTC) + timedelta(hours=6),
    )


class SearchDestinationsTool(TravelTool):
    spec = ToolSpec(
        name="search_destinations",
        description="根据出发地、预算、天数、气候和拥挤度偏好推荐目的地",
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        preferences = " ".join(arguments.get("preferences", []))
        avoid_cold = any(word in preferences for word in ["不冷", "怕冷", "不要冷", "暖和"])
        candidates = [
            {"name": "厦门", "reason": "气候相对温和，城市和海边体验结合，适合轻松旅行", "estimated_cost": 5600, "crowd_level": "中"},
            {"name": "成都", "reason": "美食、城市文化和周边自然体验丰富", "estimated_cost": 4800, "crowd_level": "中高"},
            {"name": "昆明", "reason": "气候舒适，适合作为慢节奏旅行和周边延伸的起点", "estimated_cost": 5200, "crowd_level": "中"},
            {"name": "大理", "reason": "适合慢旅行、自然风景和小众体验", "estimated_cost": 6200, "crowd_level": "中"},
            {"name": "杭州", "reason": "城市文化、湖区和周边短途景点组合方便", "estimated_cost": 5100, "crowd_level": "中高"},
        ]
        if avoid_cold:
            candidates = [item for item in candidates if item["name"] != "昆明"] + [
                {"name": "海口", "reason": "气候偏暖，适合希望避寒的旅行者", "estimated_cost": 5900, "crowd_level": "中"}
            ]
        return _result(self.spec.name, {"destinations": candidates[:4]}, "已根据当前约束生成目的地候选。")


class SearchAttractionsTool(TravelTool):
    spec = ToolSpec(
        name="search_attractions",
        description="搜索目的地景点并按照旅行偏好生成候选",
    )

    catalog = {
        "成都": [
            {"name": "宽窄巷子", "type": "城市文化", "estimated_hours": 2, "estimated_cost": 0},
            {"name": "成都大熊猫繁育研究基地", "type": "自然与动物", "estimated_hours": 4, "estimated_cost": 55},
            {"name": "武侯祠与锦里", "type": "历史文化", "estimated_hours": 3, "estimated_cost": 50},
            {"name": "青城山", "type": "自然与轻徒步", "estimated_hours": 6, "estimated_cost": 100},
        ],
        "厦门": [
            {"name": "鼓浪屿", "type": "海岛与建筑", "estimated_hours": 6, "estimated_cost": 80},
            {"name": "环岛路", "type": "海滨休闲", "estimated_hours": 3, "estimated_cost": 0},
            {"name": "沙坡尾", "type": "城市文化", "estimated_hours": 2, "estimated_cost": 0},
            {"name": "集美学村", "type": "建筑与慢游", "estimated_hours": 3, "estimated_cost": 0},
        ],
        "杭州": [
            {"name": "西湖", "type": "自然与城市", "estimated_hours": 5, "estimated_cost": 0},
            {"name": "中国茶叶博物馆", "type": "文化", "estimated_hours": 2, "estimated_cost": 0},
            {"name": "良渚古城遗址公园", "type": "历史文化", "estimated_hours": 5, "estimated_cost": 60},
            {"name": "龙井村", "type": "自然与茶文化", "estimated_hours": 3, "estimated_cost": 0},
        ],
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        destination = arguments.get("destination", "")
        attractions = self.catalog.get(destination, [
            {"name": f"{destination}城市核心区", "type": "城市体验", "estimated_hours": 3, "estimated_cost": 0},
            {"name": f"{destination}当地博物馆或历史街区", "type": "文化", "estimated_hours": 3, "estimated_cost": 30},
            {"name": f"{destination}周边自然景点", "type": "自然", "estimated_hours": 5, "estimated_cost": 80},
        ])
        return _result(self.spec.name, {"destination": destination, "attractions": attractions}, f"已找到 {destination} 的景点候选。")


class LookupWeatherTool(TravelTool):
    spec = ToolSpec(
        name="lookup_weather",
        description="提供目的地的示例天气与出行提醒；真实接入时替换为天气 API",
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        destination = arguments.get("destination", "未知目的地")
        weather = {"destination": destination, "temperature": "18-27°C", "rain_risk": "中等", "advice": "建议携带轻薄外套和折叠伞；当前为 Mock 数据。"}
        return _result(self.spec.name, weather, f"已取得 {destination} 的天气摘要。")


class SearchHotelsTool(TravelTool):
    spec = ToolSpec(
        name="search_hotels",
        description="提供目的地的住宿候选和示例价格；真实接入时替换为酒店服务",
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        destination = arguments.get("destination", "未知目的地")
        hotels = [
            {"name": f"{destination}城市中心精选酒店", "area": "城市中心", "nightly_price": 520, "rating": 4.5, "reason": "交通方便"},
            {"name": f"{destination}安静街区精品酒店", "area": "安静街区", "nightly_price": 420, "rating": 4.3, "reason": "适合慢旅行"},
            {"name": f"{destination}地铁附近经济酒店", "area": "地铁沿线", "nightly_price": 280, "rating": 4.1, "reason": "预算友好"},
        ]
        return _result(self.spec.name, {"destination": destination, "hotels": hotels}, "已生成住宿候选；价格为 Mock 数据。")


class SearchFlightsTool(TravelTool):
    spec = ToolSpec(
        name="search_flights",
        description="提供出发地到目的地的交通候选；真实接入时替换为机票服务",
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        origin = arguments.get("origin", "出发地")
        destination = arguments.get("destination", "目的地")
        options = [
            {"type": "flight", "route": f"{origin}-{destination}", "duration": "约 3 小时", "estimated_price": 900, "currency": "CNY", "note": "价格为 Mock 数据"},
            {"type": "train_or_flight", "route": f"{origin}-{destination}", "duration": "约 5-8 小时", "estimated_price": 650, "currency": "CNY", "note": "需要根据日期进一步核验"},
        ]
        return _result(self.spec.name, {"options": options}, "已生成交通候选；价格为 Mock 数据。")


class PlanRouteTool(TravelTool):
    spec = ToolSpec(
        name="plan_route",
        description="根据景点候选生成按天的顺序；真实接入时替换为地图路线服务",
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        places = arguments.get("places", [])
        days = max(int(arguments.get("duration_days", 3) or 3), 1)
        groups = [[] for _ in range(min(days, len(places)))]
        for index, place in enumerate(places):
            groups[min(index * len(groups) // len(places), len(groups) - 1)].append(place)
        route = []
        for index in range(days):
            day_places = groups[index] if index < len(groups) else []
            title = "集中安排相邻体验" if day_places else "自由安排或休息"
            route.append({"day": index + 1, "title": f"第 {index + 1} 天：{title}", "places": day_places, "transport_note": "优先按同一区域安排，减少折返；路线为 Mock 规划。"})
        return _result(self.spec.name, {"route": route}, "已生成按天路线草案。")


class CalculateBudgetTool(TravelTool):
    spec = ToolSpec(
        name="calculate_budget",
        description="汇总交通、住宿、景点和餐饮预算",
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        days = max(int(arguments.get("duration_days", 3) or 3), 1)
        travelers = max(int(arguments.get("traveler_count", 2) or 2), 1)
        hotel = float(arguments.get("hotel_nightly_price", 420)) * max(days - 1, 1)
        transport = float(arguments.get("transport_price", 900)) * travelers
        attractions = float(arguments.get("attraction_cost", 80)) * travelers
        food = 180.0 * days * travelers
        breakdown = {"transport": transport, "accommodation": hotel, "attractions": attractions, "food": food}
        return _result(self.spec.name, {"breakdown": breakdown, "total": sum(breakdown.values())}, "已完成预算估算；未包含真实实时价格波动。")


def build_mock_tools() -> list[TravelTool]:
    return [
        SearchDestinationsTool(),
        SearchAttractionsTool(),
        LookupWeatherTool(),
        SearchHotelsTool(),
        SearchFlightsTool(),
        PlanRouteTool(),
        CalculateBudgetTool(),
    ]
