from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.research.premier_league_historical_odds import (
    DEFAULT_CREDIT_RESERVE,
    DEFAULT_MAX_CREDIT_BUDGET,
    plan_historical_odds_backfill,
    run_historical_odds_backfill,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Consuma i crediti esistenti per il backfill",
    )
    parser.add_argument(
        "--max-credit-budget",
        type=int,
        default=DEFAULT_MAX_CREDIT_BUDGET,
    )
    parser.add_argument(
        "--credit-reserve",
        type=int,
        default=DEFAULT_CREDIT_RESERVE,
    )
    parser.add_argument(
        "--max-fixtures",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if args.execute:
        result = run_historical_odds_backfill(
            REPO_ROOT,
            max_credit_budget=args.max_credit_budget,
            credit_reserve=args.credit_reserve,
            max_fixtures=args.max_fixtures,
        )
    else:
        result = plan_historical_odds_backfill(
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
