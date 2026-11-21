from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from travel_agent.models import ToolResult, ToolSpec


class TravelTool(ABC):
    spec: ToolSpec

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError
