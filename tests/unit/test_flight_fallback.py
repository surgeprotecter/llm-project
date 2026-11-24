from travel_agent.graph.engine import RuntimeEngine, flight_table_lines


def lookup_state(flights):
    return {
        "request_scope": {"scenario": "lookup"},
        "tool_results": [
            {"tool_name": "google_flights__search_flights", "success": True,
             "data": {"trip_type": "round_trip", "flights": flights}},
        ],
    }


def test_evidence_reply_renders_last_successful_flight_lookup() -> None:
    state = lookup_state([
        {"price": 5610, "currency": "CNY", "primary_airline_name": "Korean Air",
         "legs": [leg("KE", "892", "2026-12-02T08:50:00", "2026-12-02T12:00:00"),
                  leg("KE", "711", "2026-12-02T13:25:00", "2026-12-02T15:55:00"),
                  leg("KE", "714", "2026-12-05T20:40:00", "2026-12-05T23:35:00"),
                  leg("KE", "881", "2026-12-06T08:25:00", "2026-12-06T09:45:00")]},
    ])

    answer = RuntimeEngine.evidence_reply(None, state)

    assert answer.startswith("自动答复校验没有通过")
    assert "KE892 / KE711" in answer
    assert "KE714 / KE881" in answer
    assert "CNY 5,610" in answer


def test_evidence_reply_skips_other_scenarios_and_failed_results() -> None:
    state = lookup_state([{"price": 1, "legs": [leg("X", "1", "2026-12-02T08:50:00", "2026-12-02T09:00:00")]}])

    assert RuntimeEngine.evidence_reply(None, {**state, "request_scope": {"scenario": "execute"}}) == ""
    failed = {**state, "tool_results": [{**state["tool_results"][0], "success": False}]}
    assert RuntimeEngine.evidence_reply(None, failed) == ""



def leg(airline_code, flight_number, departure, arrival, airline=None):
    return {
        "airline_code": airline_code,
        "flight_number": flight_number,
        "departure_time": departure,
        "arrival_time": arrival,
        "airline": airline or airline_code,
    }


def test_round_trip_rows_split_outbound_and_return() -> None:
    data = {
        "trip_type": "round_trip",
        "flights": [
            {
                "price": 5610,
                "currency": "CNY",
                "primary_airline_name": "Korean Air",
                "legs": [
                    leg("KE", "892", "2026-12-02T08:50:00", "2026-12-02T12:00:00"),
                    leg("KE", "711", "2026-12-02T13:25:00", "2026-12-02T15:55:00"),
                    leg("KE", "714", "2026-12-05T20:40:00", "2026-12-05T23:35:00"),
                    leg("KE", "881", "2026-12-06T08:25:00", "2026-12-06T09:45:00"),
                ],
            }
        ],
    }

    rows = flight_table_lines(data)

    assert rows[0].startswith("| # |")
    body = rows[2]
    assert "KE892 / KE711" in body
    assert "KE714 / KE881" in body
    assert "12-02 08:50" in body
    assert "CNY 5,610" in body


def test_one_way_keeps_all_legs_in_outbound_column() -> None:
    data = {
        "trip_type": "one_way",
        "flights": [
            {
                "price": 769,
                "currency": "CNY",
                "primary_airline_name": "Thai AirAsia",
                "legs": [
                    leg("FD", "581", "2026-12-10T02:30:00", "2026-12-10T06:10:00"),
                    leg("FD", "760", "2026-12-10T08:15:00", "2026-12-10T09:55:00"),
                ],
            }
        ],
    }

    body = flight_table_lines(data)[2]

    assert "FD581 / FD760" in body
    assert body.count("|") == 6
    assert body.split("|")[4].strip() == "—"


def test_skips_malformed_and_respects_limit() -> None:
    data = {
        "trip_type": "one_way",
        "flights": [
            {"price": 100, "currency": "CNY", "legs": []},
            *[
                {"price": 200 + index, "currency": "CNY", "primary_airline_name": "X",
                 "legs": [leg("X", str(index), "2026-12-10T08:00:00", "2026-12-10T09:00:00")]}
                for index in range(10)
            ],
        ],
    }

    rows = flight_table_lines(data, limit=3)

    assert len(rows) == 5  # header, separator, and three data rows
    assert flight_table_lines({"flights": []}) == []
