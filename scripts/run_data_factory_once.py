from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.live_provider_adapter import LiveProviderAdapter


def main() -> None:
    config = CollectorConfig(db_path=Path("data/collector.sqlite"))
    repo = CollectorRepository(config)
    adapter = LiveProviderAdapter(config)
    fixtures = adapter.fetch_fixtures()

    downloaded = len(fixtures)
    inserted = 0
    skipped = 0
    for fixture in fixtures:
        provider_fixture_id = str(fixture.get("provider_fixture_id") or "")
        if not provider_fixture_id:
            continue
        existing = repo.get_fixture(provider_fixture_id)
        if existing is None:
            repo.upsert_fixture(fixture)
            inserted += 1
        else:
            skipped += 1

    print(f"Fixtures downloaded: {downloaded}")
    print(f"Fixtures inserted: {inserted}")
    print(f"Fixtures skipped: {skipped}")


if __name__ == "__main__":
    main()
