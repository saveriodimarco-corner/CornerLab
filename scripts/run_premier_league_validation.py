from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.research.premier_league_validation import (
    run_premier_league_validation,
)


if __name__ == "__main__":
    result = run_premier_league_validation(
        base_dir=REPO_ROOT,
    )

    print(
        json.dumps(
            {
                "run_type": result.get("run_type"),
                "validation_passed": result.get(
                    "validation_passed"
                ),
                "model_name": result.get("model_name"),
                "mae": result.get("mae"),
                "baseline_mae": result.get(
                    "baseline_mae"
                ),
                "gates": result.get("gates"),
                "serie_a_artifacts_unchanged": (
                    result.get(
                        "serie_a_artifacts_unchanged"
                    )
                ),
                "published_artifacts": result.get(
                    "published_artifacts"
                ),
                "diagnostic_dataset_path": (
                    result.get(
                        "diagnostic_dataset_path"
                    )
                ),
                "report_path": result.get(
                    "report_path"
                ),
            },
            ensure_ascii=True,
        )
    )
