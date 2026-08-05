from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.result_resolver import ResultResolver


def main() -> None:
    config = CollectorConfig(db_path=Path("data/collector.sqlite"))
    repo = CollectorRepository(config)
    resolver = ResultResolver(config, repo)
    resolver.upsert_result({
        "fixture_id": 1,
        "home_score": 1,
        "away_score": 0,
        "home_corners": None,
        "away_corners": None,
        "total_corners": None,
        "settled_at": config.now_utc(),
        "provider": "manual",
    })


if __name__ == "__main__":
    main()
