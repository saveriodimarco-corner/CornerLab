from __future__ import annotations

from src.data.providers.odds.api_football_odds import ApiFootballOddsProvider


def test_find_corner_bet_catalog_entries() -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")
    payload = {
        "response": [
            {"id": 101, "name": "Over/Under Corners"},
            {"id": 102, "name": "Over/Under Goals"},
            {"id": 103, "name": "Alternative Corners"},
        ]
    }
    entries = provider.find_corner_bet_entries(payload)
    assert [entry["id"] for entry in entries] == [101, 103]
    assert [entry["name"] for entry in entries] == ["Over/Under Corners", "Alternative Corners"]


def test_extract_corner_odds_rows() -> None:
    provider = ApiFootballOddsProvider(api_key="test-key")
    payload = {
        "response": [
            {
                "bookmaker": {"name": "Bet365"},
                "bets": [
                    {
                        "id": 501,
                        "name": "Corners Over/Under",
                        "values": [
                            {"value": "8.5", "odd": "1.91"},
                            {"value": "9.5", "odd": "1.95"},
                        ],
                    }
                ],
            }
        ]
    }
    rows = provider.extract_corner_odds_rows(payload)
    assert len(rows) == 2
    assert rows[0]["bookmaker"] == "Bet365"
    assert rows[0]["bet_name"] == "Corners Over/Under"
    assert rows[0]["value"] == "8.5"
