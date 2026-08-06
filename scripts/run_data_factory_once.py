from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.live_provider_adapter import LiveProviderAdapter
from src.collector.odds_collector import OddsCollector


def main() -> None:
    config = CollectorConfig(db_path=Path("data/collector.sqlite"))
    repo = CollectorRepository(config)
    adapter = LiveProviderAdapter(config)

    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT provider_fixture_id, fixture_id FROM collector_fixtures ORDER BY fixture_id").fetchall()
    finally:
        conn.close()

    fixtures_checked = 0
    fixtures_with_odds = 0
    odds_downloaded = 0
    odds_inserted = 0
    odds_skipped = 0
    genuine_corner_inserted = 0
    odds_checked = 0
    odds_retry_skipped = 0
    odds_pending_retry = 0

    collector = OddsCollector(config, repo)

    for row in rows:
        provider_fixture_id = str(row["provider_fixture_id"] or "")
        fixture_id = row["fixture_id"]
        if not provider_fixture_id:
            continue
        fixtures_checked += 1
        result = collector.collect_odds_for_fixture(fixture_id, provider_fixture_id, lambda fixture_id_value: adapter.fetch_odds(fixture_id_value), provider="api-football")
        odds_checked += result["checked"]
        odds_downloaded += result["downloaded"]
        odds_inserted += result["inserted"]
        odds_skipped += 1 if result.get("skipped") else 0
        if result.get("skipped"):
            odds_retry_skipped += 1
        if repo.get_odds_status(fixture_id, provider="api-football") is not None:
            odds_pending_retry += 1
        if result["downloaded"]:
            fixtures_with_odds += 1

    print(f"Fixtures checked: {fixtures_checked}")
    print(f"Fixtures with odds: {fixtures_with_odds}")
    print(f"Odds checked: {odds_checked}")
    print(f"Odds skipped by TTL: {odds_retry_skipped}")
    print(f"Odds pending retry: {odds_pending_retry}")
    print(f"Odds snapshots downloaded: {odds_downloaded}")
    print(f"Odds snapshots inserted: {odds_inserted}")
    print(f"Odds snapshots skipped: {odds_skipped}")
    print(f"Genuine corner snapshots inserted: {genuine_corner_inserted}")


if __name__ == "__main__":
    main()
