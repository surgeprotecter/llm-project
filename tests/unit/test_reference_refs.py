"""Tests for accepting workspace place ids as task evidence references."""
from travel_agent.graph.engine import RuntimeEngine


class FakeStore:
    def __init__(self, results):
        self.results = results

    def result(self, thread_id, result_id):
        return self.results.get(result_id)


def engine_with(results=None):
    engine = RuntimeEngine.__new__(RuntimeEngine)
    engine.store = FakeStore(results or {})
    return engine


def state_with(**workspace):
    data = {"places": [{"id": "osm:node:579168324"}, {"id": "amap:B0001"}], "sources": []}
    data.update(workspace)
    return {"thread_id": "thread-1", "report_workspace": {"data": data}, "tool_results": []}


def test_workspace_place_id_accepted():
    assert engine_with().reference_exists(state_with(), "osm:node:579168324")


def test_stored_result_id_accepted():
    engine = engine_with({"result-1": {"result": {}}})
    assert engine.reference_exists(state_with(), "result-1")


def test_unknown_reference_rejected():
    assert not engine_with().reference_exists(state_with(), "osm:node:999")


def test_workspace_prefix_and_artifact(tmp_path):
    engine = engine_with()
    state = state_with(days=[{"day": 1}])
    assert engine.reference_exists(state, "workspace:days")
    assert not engine.reference_exists(state, "workspace:transport")
    artifact = tmp_path / "report.html"
    artifact.write_text("<html></html>", encoding="utf-8")
    assert engine.reference_exists(state, f"artifact:{artifact}")


def test_reference_hints_list_places_then_results():
    engine = engine_with()
    state = state_with()
    state["tool_results"] = [{"result_id": "result-9"}]
    hints = engine.reference_hints(state)
    assert "osm:node:579168324" in hints
    assert "result-9" in hints
