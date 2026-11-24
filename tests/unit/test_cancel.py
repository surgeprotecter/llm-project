"""Cooperative cancellation: the bridge can stop the turn it actually started."""
from __future__ import annotations

import threading
from types import SimpleNamespace

from travel_agent.server import AgentRuntimeService


class FakeGateway:
    def __init__(self) -> None:
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1


class FakeRegistry:
    def __init__(self) -> None:
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1


class FakeGraph:
    def __init__(self) -> None:
        self.gateway = FakeGateway()
        self.registry = FakeRegistry()

    def get_state(self, config):  # pragma: no cover - trivially delegates
        return SimpleNamespace(next=None, values={})


class CancellingGraph(FakeGraph):
    def invoke(self, value, config=None, **kwargs):
        raise KeyboardInterrupt


class FakeDatabase:
    def latest_report(self, trip_id):
        return None


def make_service(active: str | None = None, graph=None) -> AgentRuntimeService:
    service = AgentRuntimeService.__new__(AgentRuntimeService)
    service._lock = threading.Lock()
    service._active_thread = active
    service._cancel_requested = False
    service._session_reports = {}
    service._graph = graph or FakeGraph()
    service._database = FakeDatabase()
    service._ensure_initialized = lambda real_only=False: None  # type: ignore[method-assign]
    return service


def test_cancel_matching_session_flags_gateway_and_registry():
    service = make_service(active="trip-a")

    assert service.cancel("trip-a") is True
    assert service._cancel_requested is True
    assert service._graph.gateway.cancel_calls == 1
    assert service._graph.registry.cancel_calls == 1


def test_cancel_other_session_does_not_touch_running_turn():
    service = make_service(active="trip-a")

    assert service.cancel("trip-b") is False
    assert service.cancel("") is False
    assert service._cancel_requested is False
    assert service._graph.gateway.cancel_calls == 0
    assert service._graph.registry.cancel_calls == 0


def test_cancel_without_running_turn_is_a_noop():
    service = make_service(active=None)

    assert service.cancel("trip-a") is False
    assert service._graph.gateway.cancel_calls == 0


def test_interrupted_turn_returns_cancelled_payload():
    service = make_service(active=None, graph=CancellingGraph())

    payload = service.run_turn("帮我做一份东京行程", session_id="trip-a", real_only=True)

    assert payload["status"] == "cancelled"
    assert "已取消本轮执行" in payload["reply"]
    assert payload["session_id"] == "trip-a"
    assert payload["report"] is None
    assert service._active_thread is None
