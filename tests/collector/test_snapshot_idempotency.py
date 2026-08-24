from pathlib import Path

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository


def test_exact_duplicate_snapshot_is_idempotent(tmp_path: Path) -> None:
    config = CollectorConfig(
        db_path=tmp_path / "collector.sqlite",
        data_dir=tmp_path,
        reports_dir=tmp_path / "reports",
        docs_dir=tmp_path / "docs",
    )

    repo = CollectorRepository(config)

    payload = {
        "fixture_id": 1,
        "bookmaker": "TESTBOOK",
        "market": "TOTAL_CORNERS_OVER",
        "line": "9.5",
        "side": "OVER",
        "decimal_odds": 1.91,
        "snapshot_timestamp": "2026-08-23T09:00:00Z",
        "minutes_to_kickoff": 60,
        "provider": "test-provider",
        "provider_event_id": "fixture-1",
        "raw_response_hash": "test",
        "import_timestamp": "2026-08-23T09:00:00Z",
    }

    first = repo.store_snapshot(payload)
    second = repo.store_snapshot(payload)

    assert first is not None
    assert second is None
