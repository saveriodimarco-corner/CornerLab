from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.scheduler import CollectorScheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = CollectorConfig(db_path=Path("data/collector.sqlite"))
    repo = CollectorRepository(config)
    scheduler = CollectorScheduler(config, repo)
    result = scheduler.run(mode="DRY_RUN" if args.dry_run else "ONE_SHOT")
    print(result)


if __name__ == "__main__":
    main()
