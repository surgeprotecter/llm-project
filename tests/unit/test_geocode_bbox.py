import time

import pytest

from travel_agent.graph.engine import issue_text
from travel_agent.tools.live_http import LiveHTTPTool, _center_pair, country_matches

HONG_KONG = (114.1372, 22.2839)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, url="https://photon.komoot.io/api/"):
        self.status_code = status_code
        self._payload = payload or {"features": []}
        self.url = url

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None):
        self.calls.append(dict(params or {}))
        return self.responses.pop(0)


def geocode_tool():
    return LiveHTTPTool(
        "open_maps__search_places",
        "geocode",
        {"query": {"type": "string"}, "country_code": {"type": "string"}, "center": {"type": "string"}},
        ["query", "country_code"],
        "geocode",
    )


def test_center_accepts_both_orders():
    assert _center_pair("114.1372,22.2839") == pytest.approx(HONG_KONG)
    assert _center_pair("22.2839,114.1372") == pytest.approx(HONG_KONG)


def test_center_rejects_garbage():
    with pytest.raises(ValueError):
        _center_pair("hong kong")
    with pytest.raises(ValueError):
        _center_pair("200,95")


def test_geocode_builds_valid_bbox_for_latitude_first_center():
    LiveHTTPTool._last_geocode = time.monotonic()
    client = FakeClient([FakeResponse()])
    data, _ = geocode_tool().geocode(client, "University of Hong Kong", "HK", "22.2839,114.1372")
    assert data["places"] == []
    assert client.calls[0]["bbox"] == "112.9372,21.0839,115.3372,23.4839"


def test_geocode_retries_without_bbox_on_400():
    LiveHTTPTool._last_geocode = time.monotonic()
    client = FakeClient([FakeResponse(400), FakeResponse()])
    data, _ = geocode_tool().geocode(client, "University of Hong Kong", "HK", "22.2839,114.1372")
    assert data["places"] == []
    assert "bbox" in client.calls[0]
    assert "bbox" not in client.calls[1]


def test_hong_kong_features_match_under_mainland_code():
    assert country_matches("CN", "HK")
    assert country_matches("HK", "HK")
    assert country_matches("cn", "mo")
    assert not country_matches("JP", "HK")
    assert country_matches("JP", "JP")


def test_geocode_uses_cn_filter_and_hk_bounds_without_center():
    LiveHTTPTool._last_geocode = time.monotonic()
    client = FakeClient([FakeResponse()])
    geocode_tool().geocode(client, "HKU", "HK")
    assert client.calls[0]["countrycode"] == "CN"
    assert client.calls[0]["bbox"] == "113.8200,22.1400,114.4400,22.5700"


def test_geocode_drops_mainland_places_for_hong_kong():
    LiveHTTPTool._last_geocode = time.monotonic()
    payload = {
        "features": [
            {
                "properties": {"osm_type": "R", "osm_id": 1, "name": "HKU", "countrycode": "CN"},
                "geometry": {"coordinates": [114.1378, 22.2828]},
            },
            {
                "properties": {"osm_type": "W", "osm_id": 2, "name": "Beijing", "countrycode": "CN"},
                "geometry": {"coordinates": [116.4, 39.9]},
            },
        ]
    }
    data, _ = geocode_tool().geocode(FakeClient([FakeResponse(payload=payload)]), "HKU", "HK")
    assert [p["name"] for p in data["places"]] == ["HKU"]


def test_geocode_keeps_hong_kong_places_reported_as_cn():
    LiveHTTPTool._last_geocode = time.monotonic()
    payload = {
        "features": [
            {
                "properties": {
                    "osm_type": "R",
                    "osm_id": 6721848,
                    "name": "The University of Hong Kong",
                    "countrycode": "CN",
                    "osm_value": "university",
                },
                "geometry": {"coordinates": [114.1378586, 22.2828768]},
            }
        ]
    }
    data, _ = geocode_tool().geocode(FakeClient([FakeResponse(payload=payload)]), "HKU", "HK")
    assert [p["name"] for p in data["places"]] == ["The University of Hong Kong"]
    assert data["places"][0]["coordinate_system"] == "WGS84"


def test_issue_text_strips_envelope_before_urls():
    raw = (
        '{"items": "API 调用失败：UNKNOWN_ERROR"}；Client error \'400 Bad Request\' for url '
        "'https://photon.komoot.io/api/?q=University+of+Hong+Kong'；Client error '400 Bad Request' for url "
        "'https://photon.komoot.io/api/?q=HKU'"
    )
    text = issue_text(raw)
    assert "API 调用失败" not in text
    assert "https://" not in text
    assert "300" not in text
    assert text.startswith("接口返回 400 Bad Request")
    assert len(text) <= 220


def test_issue_text_keeps_plain_message():
    assert issue_text("  未找到符合条件的地点；  ") == "未找到符合条件的地点"
    assert issue_text("") == "步骤执行时遇到接口错误"
