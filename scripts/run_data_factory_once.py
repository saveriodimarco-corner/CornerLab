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


def _apply_result_counters(
    result: dict,
    odds_checked: int,
    odds_downloaded: int,
    odds_inserted: int,
    odds_skipped: int,
    odds_retry_skipped: int,
    genuine_corner_inserted: int,
) -> tuple[int, int, int, int, int, int]:
    odds_checked += int(result.get("checked") or 0)
    odds_downloaded += int(result.get("downloaded") or 0)
    inserted = int(result.get("inserted") or 0)
    odds_inserted += inserted
    # The live The Odds API adapter is already constrained to genuine total-corner rows.
    genuine_corner_inserted += inserted
    skipped = bool(result.get("skipped"))
    odds_skipped += 1 if skipped else 0
    odds_retry_skipped += 1 if skipped else 0
    return odds_checked, odds_downloaded, odds_inserted, odds_skipped, odds_retry_skipped, genuine_corner_inserted


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
        result = collector.collect_odds_for_fixture(fixture_id, provider_fixture_id, lambda fixture_id_value: adapter.fetch_odds(fixture_id_value), provider="the-odds-api")
        odds_checked, odds_downloaded, odds_inserted, odds_skipped, odds_retry_skipped, genuine_corner_inserted = _apply_result_counters(
            result,
            odds_checked,
            odds_downloaded,
            odds_inserted,
            odds_skipped,
            odds_retry_skipped,
            genuine_corner_inserted,
        )
        if repo.get_odds_status(fixture_id, provider="the-odds-api") is not None:
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
