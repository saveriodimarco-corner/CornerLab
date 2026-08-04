from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare historical The Odds API download plan")
    parser.add_argument("--execute", action="store_true", help="Execute the historical download plan")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--snapshot-offset-minutes", type=int, default=10)
    parser.add_argument("--reserve-credits", type=int, default=100)
    args = parser.parse_args()

    if not args.execute:
        print("DRY RUN: no paid historical requests will be sent")
        print(f"Start date: {args.start_date}")
        print(f"End date: {args.end_date}")
        print(f"Snapshot offset minutes: {args.snapshot_offset_minutes}")
        print(f"Reserve credits: {args.reserve_credits}")
        return 0

    if not os.getenv("THE_ODDS_API_KEY"):
        raise SystemExit("THE_ODDS_API_KEY is required for execution")

    print("Historical execution is not implemented in this sprint; this stub requires explicit --execute")
    return 0
