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
