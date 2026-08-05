from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.collector_health import CollectorHealth


def main() -> None:
    config = CollectorConfig(db_path=Path("data/collector.sqlite"))
    repo = CollectorRepository(config)
    health = CollectorHealth(config, repo)
    print(health.build_report())


if __name__ == "__main__":
    main()
