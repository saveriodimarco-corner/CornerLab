from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import import_data


def test_import_data_creates_reports_and_db(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    import_data.main()

    assert (tmp_path / "reports" / "quality_report.md").exists()
    assert (tmp_path / "data" / "raw" / "acquisition.db").exists()
