from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.research.premier_league_2024_25_backfill import (
    plan_2024_25_backfill,
    run_2024_25_backfill,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
    )
    parser.add_argument(
        "--max-fixtures",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if args.execute:
        result = run_2024_25_backfill(
            REPO_ROOT,
            max_fixtures=args.max_fixtures,
        )
    else:
        result = plan_2024_25_backfill(
            REPO_ROOT
        )

    print(
        json.dumps(
            result,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
