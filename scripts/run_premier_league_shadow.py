from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.operations.premier_league_forward_notifier import (
    notify_forward_price_checks,
)
from src.operations.premier_league_forward_settlement import (
    run_forward_settlement,
)
from src.operations.premier_league_shadow import (
    run_premier_league_shadow,
)
from src.research.premier_league_forward_scoring import (
    run_forward_scoring,
)


def run_pipeline(
    base_dir: Path | str = REPO_ROOT,
) -> dict:
    base_dir = Path(base_dir)

    collection = run_premier_league_shadow(
        base_dir=base_dir,
        output_dir=base_dir,
    )
    settlement = run_forward_settlement(
        base_dir
    )
    scoring = run_forward_scoring(
        base_dir
    )
    notifications = (
        notify_forward_price_checks(
            base_dir,
            dry_run=False,
        )
    )

    return {
        "run_type": (
            "premier_league_forward_pipeline"
        ),
        "mode": "forward_validation_only",
        "competition": "Premier League",
        "collection": {
            key: collection.get(key)
            for key in [
                "started_at",
                "completed_at",
                "fixtures_fetched",
                "odds_refresh_attempts",
                "odds_downloaded",
                "odds_writes",
                "duplicates_removed",
                "conflicting_duplicates",
                "quota_remaining",
                "provider_status",
                "database_path",
                "report_path",
            ]
        },
        "settlement": settlement,
        "scoring": scoring,
        "notifications": notifications,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            run_pipeline(),
            ensure_ascii=True,
        )
    )
