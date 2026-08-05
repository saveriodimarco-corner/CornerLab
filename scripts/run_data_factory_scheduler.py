from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.scheduler import CollectorScheduler


def main() -> None:
    config = CollectorConfig(db_path=Path("data/collector.sqlite"))
    repo = CollectorRepository(config)
    scheduler = CollectorScheduler(config, repo)
    while True:
        scheduler.run(mode="ONE_SHOT")
        time.sleep(config.scheduler_interval_minutes * 60)


if __name__ == "__main__":
    main()
