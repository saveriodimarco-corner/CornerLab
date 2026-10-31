from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.operations.premier_league_forward_notifier import (
    notify_forward_price_checks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--send",
        action="store_true",
        help=(
            "Invia gli avvisi esclusivamente "
            "in modalità Telegram simulata"
        ),
    )
    args = parser.parse_args()

    result = notify_forward_price_checks(
        REPO_ROOT,
        dry_run=not args.send,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
