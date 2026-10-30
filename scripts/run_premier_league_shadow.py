from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.operations.premier_league_shadow import (
    run_premier_league_shadow,
)


if __name__ == "__main__":
    result = run_premier_league_shadow(
        base_dir=REPO_ROOT,
        output_dir=REPO_ROOT,
    )

    print(
        json.dumps(
            {
                key: result.get(key)
                for key in [
                    "run_type",
                    "mode",
                    "competition",
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
            ensure_ascii=True,
        )
    )
