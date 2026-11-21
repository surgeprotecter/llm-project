from travel_agent.tools.normalization import normalize_travel_data


def test_flight_results_are_normalized_without_unstructured_placeholders() -> None:
    result = normalize_travel_data(
        "google_flights__search_flights",
        "google_flights",
        {
            "flights": [
                {
                    "price": 973,
                    "currency": "CNY",
                    "primary_airline_name": "Hong Kong Express",
                    "legs": [
                        {
                            "departure_airport_name": "Hong Kong International Airport",
                            "arrival_airport_name": "Danang International Airport",
                            "departure_time": "2026-12-10T16:50:00",
                            "arrival_time": "2026-12-10T17:45:00",
                            "duration": 115,
                            "flight_number": "558",
                        }
                    ],
                },
                {"price": 0},
            ]
        },
    )

    assert len(result["options"]) == 1
    option = result["options"][0]
    assert "Hong Kong International Airport → Danang International Airport" in option["route"]
    assert option["flightNumber"] == "558"
    assert option["durationText"] == "16:50–17:45 · 115 分钟"
    assert option["estimated_price"] == 973
    assert option["currency"] == "CNY"


def test_amap_poi_and_route_results_are_normalized() -> None:
    places = normalize_travel_data(
        "amap_maps__maps_text_search",
        "amap_maps",
        {"pois": [{"id": "poi-1", "name": "宽窄巷子", "location": "104.06,30.67"}]},
    )
    route = normalize_travel_data(
        "amap_maps__maps_direction_driving",
        "amap_maps",
        {"paths": [{"distance": "1200", "duration": "600"}]},
    )

    assert places["attractions"][0]["name"] == "宽窄巷子"
    assert route["route"][0]["distance"] == "1200"
