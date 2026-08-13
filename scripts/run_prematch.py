from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.operations.prematch_runner import run_prematch


def main() -> None:
    result = run_prematch(base_dir=REPO_ROOT, output_dir=REPO_ROOT, bankroll=100.0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
