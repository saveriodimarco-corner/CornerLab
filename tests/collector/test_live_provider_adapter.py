from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.live_provider_adapter import LiveProviderAdapter


class _FakeOddsApi:
    def __init__(self, events, odds_rows):
        self._events = events
        self._odds_rows = odds_rows

    def list_events(self, sport: str | None = None):
        return self._events

    def fetch_event_odds(self, event_id: str | None = None, sport: str | None = None):
        import pandas as pd

        return pd.DataFrame(self._odds_rows)


class _FakeApiFootball:
    def __init__(self, payloads_by_league):
        self.payloads_by_league = payloads_by_league

    def _perform_request(self, path: str, params=None):
        league_id = int((params or {}).get('league'))
        return {'response': self.payloads_by_league.get(league_id, [])}


def test_fetch_odds_skips_unresolved_fixture() -> None:
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'collector.sqlite'
        config = CollectorConfig(db_path=db_path)
        CollectorRepository(config)
        adapter = LiveProviderAdapter(config)
        adapter.the_odds_api = _FakeOddsApi(events=[], odds_rows=[])

        rows = adapter.fetch_odds('missing-fixture-id')

        assert rows == []
        assert adapter.last_odds_resolution['missing-fixture-id']['match_status'] == 'UNMATCHED'


def test_fetch_odds_uses_matched_the_odds_event_id() -> None:
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'collector.sqlite'
        config = CollectorConfig(db_path=db_path)
        repo = CollectorRepository(config)
        repo.upsert_fixture(
            {
                'provider_fixture_id': 'api-football-1',
                'competition': 'Serie A',
                'season': '2026',
                'kickoff_utc': '2026-08-20T18:45:00Z',
                'home_team': 'Juventus',
                'away_team': 'Inter',
                'status': 'NS',
                'provider': 'api-football',
            }
        )

        adapter = LiveProviderAdapter(config)
        adapter.the_odds_api = _FakeOddsApi(
            events=[
                {
                    'id': 'evt-abc',
                    'home_team': 'Juventus',
                    'away_team': 'Inter',
                    'commence_time': '2026-08-20T19:00:00Z',
                }
            ],
            odds_rows=[
                {
                    'bookmaker': 'BetRivers',
                    'market': 'TOTAL_CORNERS_OVER',
                    'line': '9.5',
                    'side': 'OVER',
                    'closing_odds': 2.1,
                }
            ],
        )

        rows = adapter.fetch_odds('api-football-1')

        assert rows
        assert rows[0]['source_fixture_id'] == 'evt-abc'
        assert rows[0]['line'] == '9.5'
        assert rows[0]['side'] == 'OVER'


def test_fetch_odds_matches_inter_alias_to_inter_milan() -> None:
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'collector.sqlite'
        config = CollectorConfig(db_path=db_path)
        repo = CollectorRepository(config)
        repo.upsert_fixture(
            {
                'provider_fixture_id': 'api-football-inter-monza',
                'competition': 'Serie A',
                'season': '2026',
                'kickoff_utc': '2026-08-22T16:30:00Z',
                'home_team': 'Inter',
                'away_team': 'Monza',
                'status': 'NS',
                'provider': 'api-football',
            }
        )

        adapter = LiveProviderAdapter(config)
        adapter.the_odds_api = _FakeOddsApi(
            events=[
                {
                    'id': 'evt-inter-milan',
                    'home_team': 'Inter Milan',
                    'away_team': 'Monza',
                    'commence_time': '2026-08-22T16:30:00Z',
                }
            ],
            odds_rows=[
                {
                    'bookmaker': 'BetRivers',
                    'market': 'TOTAL_CORNERS_OVER',
                    'line': '9.5',
                    'side': 'OVER',
                    'closing_odds': 2.02,
                }
            ],
        )

        rows = adapter.fetch_odds('api-football-inter-monza')

        assert rows
        assert rows[0]['source_fixture_id'] == 'evt-inter-milan'


def test_fetch_odds_matches_atalanta_alias_to_atalanta_bc() -> None:
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'collector.sqlite'
        config = CollectorConfig(db_path=db_path)
        repo = CollectorRepository(config)
        repo.upsert_fixture(
            {
                'provider_fixture_id': 'api-football-atalanta-sassuolo',
                'competition': 'Serie A',
                'season': '2026',
                'kickoff_utc': '2026-08-23T18:45:00Z',
                'home_team': 'Atalanta',
                'away_team': 'Sassuolo',
                'status': 'NS',
                'provider': 'api-football',
            }
        )

        adapter = LiveProviderAdapter(config)
        adapter.the_odds_api = _FakeOddsApi(
            events=[
                {
                    'id': 'evt-atalanta-bc',
                    'home_team': 'Atalanta BC',
                    'away_team': 'Sassuolo',
                    'commence_time': '2026-08-23T18:45:00Z',
                }
            ],
            odds_rows=[
                {
                    'bookmaker': 'BetRivers',
                    'market': 'TOTAL_CORNERS_UNDER',
                    'line': '10.5',
                    'side': 'UNDER',
                    'closing_odds': 1.90,
                }
            ],
        )

        rows = adapter.fetch_odds('api-football-atalanta-sassuolo')

        assert rows
        assert rows[0]['source_fixture_id'] == 'evt-atalanta-bc'


def test_fetch_fixtures_collects_serie_a_and_premier_league() -> None:
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'collector.sqlite'
        config = CollectorConfig(db_path=db_path)
        CollectorRepository(config)
        adapter = LiveProviderAdapter(config)
        adapter.api_football = _FakeApiFootball(
            {
                135: [
                    {
                        'fixture': {'id': 1, 'date': '2026-08-20T18:45:00Z', 'status': {'short': 'NS'}},
                        'teams': {'home': {'name': 'Inter'}, 'away': {'name': 'Roma'}},
                    }
                ],
                136: [
                    {
                        'fixture': {'id': 2, 'date': '2026-08-21T18:45:00Z', 'status': {'short': 'NS'}},
                        'teams': {'home': {'name': 'Palermo'}, 'away': {'name': 'Bari'}},
                    }
                ],
                39: [
                    {
                        'fixture': {'id': 3, 'date': '2026-08-22T18:45:00Z', 'status': {'short': 'NS'}},
                        'teams': {'home': {'name': 'Arsenal'}, 'away': {'name': 'Chelsea'}},
                    }
                ],
            }
        )

        rows = adapter.fetch_fixtures()

        assert len(rows) == 2
        assert {row['competition'] for row in rows} == {'Serie A', 'Premier League'}


def test_fetch_odds_uses_epl_sport_key_for_premier_league_fixture() -> None:
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'collector.sqlite'
        config = CollectorConfig(db_path=db_path)
        repo = CollectorRepository(config)
        repo.upsert_fixture(
            {
                'provider_fixture_id': 'api-football-serie-b-1',
                'competition': 'Premier League',
                'season': '2026',
                'kickoff_utc': '2026-08-20T18:45:00Z',
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'status': 'NS',
                'provider': 'api-football',
            }
        )

        captured = {}

        class _EplOddsApi(_FakeOddsApi):
            def list_events(self, sport: str | None = None):
                captured['list_sport'] = sport
                return super().list_events(sport=sport)

            def fetch_event_odds(self, event_id: str | None = None, sport: str | None = None):
                captured['odds_sport'] = sport
                return super().fetch_event_odds(event_id=event_id, sport=sport)

        adapter = LiveProviderAdapter(config)
        adapter.the_odds_api = _EplOddsApi(
            events=[
                {
                    'id': 'evt-epl',
                    'home_team': 'Arsenal',
                    'away_team': 'Chelsea',
                    'commence_time': '2026-08-20T18:45:00Z',
                }
            ],
            odds_rows=[
                {
                    'bookmaker': 'BetRivers',
                    'market': 'TOTAL_CORNERS_OVER',
                    'line': '9.5',
                    'side': 'OVER',
                    'closing_odds': 2.1,
                }
            ],
        )

        rows = adapter.fetch_odds('api-football-serie-b-1')

        assert rows
        assert captured['list_sport'] == 'soccer_epl'
        assert captured['odds_sport'] == 'soccer_epl'
