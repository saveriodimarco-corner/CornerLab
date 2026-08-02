from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.provenance import verify_provenance


def test_provenance_verification_generates_report_and_manifest(tmp_path) -> None:
    report_path, manifest_path = verify_provenance(tmp_path)

    assert report_path.exists()
    assert manifest_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "# Data Provenance Verification" in report_text
    assert "2023/24" in report_text

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 3
    assert {"season", "source_file_name", "source_url", "source_row_count", "file_sha256", "import_timestamp"}.issubset(manifest.columns)


def test_provenance_verification_matches_database_rows(tmp_path) -> None:
    report_path, manifest_path = verify_provenance(tmp_path)
    manifest = pd.read_csv(manifest_path)

    assert (manifest["source_row_count"] == manifest["db_row_count"]).all()
    assert (manifest["team_names_match"] == True).all()
    assert (manifest["dates_match"] == True).all()
    assert (manifest["corners_match"] == True).all()
    assert (manifest["synthetic_fixture_count"] == 0).all()
    assert (manifest["test_fixture_count"] == 0).all()

    report_text = report_path.read_text(encoding="utf-8")
    assert "https://www.football-data.co.uk/mmz4281/2324/I1.csv" in report_text
    assert "https://www.football-data.co.uk/mmz4281/2425/I1.csv" in report_text
    assert "https://www.football-data.co.uk/mmz4281/2526/I1.csv" in report_text
