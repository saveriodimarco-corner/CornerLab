from src.research.paper_trading import (
    LIVE_TEAM_ALIASES,
)


def test_live_premier_league_names_match_football_data_history():
    expected = {
        "Brighton and Hove Albion": "Brighton",
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle United": "Newcastle",
        "Nottingham Forest": "Nott'm Forest",
        "Tottenham Hotspur": "Tottenham",
        "Coventry City": "Coventry",
        "Hull City": "Hull",
        "Ipswich Town": "Ipswich",
        "Leeds United": "Leeds",
    }

    for live_name, historical_name in expected.items():
        assert LIVE_TEAM_ALIASES[live_name] == historical_name
