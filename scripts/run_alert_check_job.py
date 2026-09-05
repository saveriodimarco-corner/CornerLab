from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.operations.automation import run_alert_check_job


if __name__ == "__main__":
    exit_code, payload = run_alert_check_job(REPO_ROOT)
    print(json.dumps(payload, ensure_ascii=True))
    raise SystemExit(exit_code)
