from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_health_check(base_dir: Path | str | None = None, output_dir: Path | str | None = None) -> Dict[str, Any]:
    base_dir = Path(base_dir) if base_dir is not None else REPO_ROOT
    output_dir = Path(output_dir) if output_dir is not None else base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, Any]] = []
    database_path = base_dir / "data" / "collector.sqlite"
    ok = True

    if database_path.exists():
        checks.append({"name": "database", "ok": True, "detail": f"Found database at {database_path}"})
    else:
        ok = False
        checks.append({"name": "database", "ok": False, "detail": f"Missing database at {database_path}"})

    reports_dir = output_dir / "reports"
    if reports_dir.exists():
        checks.append({"name": "reports", "ok": True, "detail": f"Found reports directory at {reports_dir}"})
    else:
        ok = False
        checks.append({"name": "reports", "ok": False, "detail": f"Missing reports directory at {reports_dir}"})

    result = {"ok": ok, "checks": checks}
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "health_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
