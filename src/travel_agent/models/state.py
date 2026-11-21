from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class TravelDates(BaseModel):
    start_date: date | None = None
    end_date: date | None = None


class Budget(BaseModel):
    # These three values always describe the whole group. A lower bound is
    # not a target, and an absent maximum means there is no stated ceiling.
    amount: float | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    currency: str = "CNY"
    includes_transport: bool = True
    includes_accommodation: bool = True
    per_person: dict[str, float | None] = Field(default_factory=dict)

    @property
    def specified(self) -> bool:
        return any(value is not None for value in (self.amount, self.min_amount, self.max_amount)) or any(
            value is not None for value in self.per_person.values()
        )

    def normalize_for_travelers(self, travelers: int | None) -> None:
        if self.per_person:
            for field in ("amount", "min_amount", "max_amount"):
                value = self.per_person.get(field)
                setattr(self, field, value * travelers if value is not None and travelers else None)


class TripState(BaseModel):
    user_id: str = "default"
    trip_id: str = "default"
    thread_id: str = "default"
    origin: str | None = None
    destinations: list[str] = Field(default_factory=list)
    travel_dates: TravelDates = Field(default_factory=TravelDates)
    duration_days: int | None = None
    traveler_count: int | None = None
    budget: Budget = Field(default_factory=Budget)
    preferences: dict[str, Any] = Field(default_factory=dict)
    hard_constraints: list[str] = Field(default_factory=list)
    confirmed_decisions: list[str] = Field(default_factory=list)
    rejected_options: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    current_phase: str = "intake"
    scenario: Literal["destination_discovery", "known_destination", "general_planner", "unknown"] = "unknown"
    selected_places: list[dict[str, Any]] = Field(default_factory=list)
    hotel_options: list[dict[str, Any]] = Field(default_factory=list)
    transport_options: list[dict[str, Any]] = Field(default_factory=list)
    route_plan: list[dict[str, Any]] = Field(default_factory=list)
    budget_breakdown: dict[str, float] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    report_version: int = 0
    conversation_summary: str = ""

    @property
    def destination(self) -> str | None:
        return self.destinations[0] if self.destinations else None

    def to_context_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("thread_id", None)
        data.pop("conversation_summary", None)
        return data
