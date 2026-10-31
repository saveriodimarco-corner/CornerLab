from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.research.premier_league_forward_scoring import (
    run_forward_scoring,
)


if __name__ == "__main__":
    result = run_forward_scoring(REPO_ROOT)
    print(
        json.dumps(
            result,
            ensure_ascii=True,
        )
    )
