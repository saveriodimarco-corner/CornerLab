from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from src.operations.automation import run_settlement_job


if __name__ == "__main__":
	exit_code, payload = run_settlement_job(REPO_ROOT)
	print(json.dumps({key: payload.get(key) for key in ["job_id", "job_type", "outcome", "started_at", "completed_at", "error"]}, ensure_ascii=True))
	raise SystemExit(exit_code)