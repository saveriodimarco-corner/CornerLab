from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data.normalizer import normalize_team_name
from src.data.provenance import verify_provenance
from src.engine.feature_store import FeatureStore


OFFICIAL_SOURCE_FILES = {
    "2023/24": "data/raw/football_data/I1_2324.csv",
    "2024/25": "data/raw/football_data/I1_2425.csv",
    "2025/26": "data/raw/football_data/I1_2526.csv",
}

CACHE_VERSION = "historical-foundation-v1"
CACHE_ROOT_RELATIVE_PATH = Path("data") / "research" / "cache" / "historical_foundation"
CACHE_ARTIFACTS = (
    Path("data/raw/serie_a_matches.csv"),
    Path("data/processed/serie_a_matches.parquet"),
    Path("data/raw/serie_a_historical.db"),
    Path("data/research/research_dataset.parquet"),
    Path("reports/database_validation.md"),
    Path("reports/descriptive_statistics.md"),
    Path("reports/total_corners_histogram.html"),
    Path("reports/total_corners_qq.html"),
    Path("reports/corner_correlation_matrix.html"),
    Path("reports/data_provenance.md"),
    Path("data/processed/source_manifest.csv"),
    Path("docs/DATA_DICTIONARY.md"),
)


def build_historical_database(base_dir: Path | None = None) -> Tuple[Path, Path, Path, Path, Path]:
    base_dir = base_dir or Path.cwd()
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    research_dir = base_dir / "data" / "research"
    reports_dir = base_dir / "reports"
    docs_dir = base_dir / "docs"
    cache_dir = base_dir / CACHE_ROOT_RELATIVE_PATH
    for directory in [raw_dir, processed_dir, research_dir, reports_dir, docs_dir, cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    fingerprint = build_historical_fingerprint(base_dir)
    stats = load_cache_stats(cache_dir)
    metadata_path = cache_dir / "metadata.json"
    metadata = load_cache_metadata(metadata_path)
    cache_hit = False

    if metadata_path.exists() and metadata.get("fingerprint") == fingerprint and metadata.get("cache_version") == CACHE_VERSION:
        cache_artifact_paths = [cache_dir / artifact.as_posix() for artifact in CACHE_ARTIFACTS]
        if all(path.exists() for path in cache_artifact_paths):
            restore_cached_artifacts(base_dir, cache_dir)
            cache_hit = True

    if cache_hit:
        start_time = time.perf_counter()
        restore_cached_artifacts(base_dir, cache_dir)
        stats["cache_hits"] += 1
        stats["last_cache_hit_seconds"] = round(time.perf_counter() - start_time, 4)
        stats["last_cache_status"] = "cache_hit"
        stats["last_cache_fingerprint"] = fingerprint
        write_cache_stats(cache_dir, stats)
        metadata = {
            "cache_version": CACHE_VERSION,
            "cache_status": "cache_hit",
            "fingerprint": fingerprint,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifacts": [artifact.as_posix() for artifact in CACHE_ARTIFACTS],
            "source_files": gather_source_file_manifest(base_dir),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        write_research_cache_report(base_dir, stats, metadata)
        return (
            base_dir / "data" / "raw" / "serie_a_matches.csv",
            base_dir / "data" / "processed" / "serie_a_matches.parquet",
            base_dir / "data" / "raw" / "serie_a_historical.db",
            base_dir / "data" / "research" / "research_dataset.parquet",
            base_dir / "reports" / "database_validation.md",
        )

    start_time = time.perf_counter()
    matches_df = generate_serie_a_matches(base_dir)
    csv_path = raw_dir / "serie_a_matches.csv"
    matches_df.to_csv(csv_path, index=False)

    canonical_path = processed_dir / "serie_a_matches.parquet"
    matches_df.to_parquet(canonical_path, index=False)

    db_path = raw_dir / "serie_a_historical.db"
    create_historical_sqlite(matches_df, db_path)

    research_df = build_research_dataset(matches_df)
    research_path = research_dir / "research_dataset.parquet"
    research_df.to_parquet(research_path, index=False)

    validation_report = build_validation_report(matches_df)
    validation_path = reports_dir / "database_validation.md"
    validation_path.write_text(validation_report, encoding="utf-8")

    descriptive_report, descriptive_artifacts = build_descriptive_report(matches_df, reports_dir)
    descriptive_path = reports_dir / "descriptive_statistics.md"
    descriptive_path.write_text(descriptive_report, encoding="utf-8")

    dictionary_path = docs_dir / "DATA_DICTIONARY.md"
    dictionary_path.write_text(build_data_dictionary(research_df), encoding="utf-8")

    verify_provenance(base_dir)

    cache_artifacts = [
        csv_path,
        canonical_path,
        db_path,
        research_path,
        validation_path,
        descriptive_path,
        *descriptive_artifacts,
        reports_dir / "data_provenance.md",
        processed_dir / "source_manifest.csv",
        dictionary_path,
    ]
    persist_cached_artifacts(base_dir, cache_dir, cache_artifacts)

    stats["rebuilds"] += 1
    stats["last_rebuild_seconds"] = round(time.perf_counter() - start_time, 4)
    stats["last_cache_status"] = "rebuilt"
    stats["last_cache_fingerprint"] = fingerprint
    write_cache_stats(cache_dir, stats)
    metadata = {
        "cache_version": CACHE_VERSION,
        "cache_status": "rebuilt",
        "fingerprint": fingerprint,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": [artifact.as_posix() for artifact in CACHE_ARTIFACTS],
        "source_files": gather_source_file_manifest(base_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_research_cache_report(base_dir, stats, metadata)

    return csv_path, canonical_path, db_path, research_path, validation_path


def generate_serie_a_matches(base_dir: Path | None = None) -> pd.DataFrame:
    base_dir = base_dir or Path.cwd()
    repo_root = Path(__file__).resolve().parents[2]
    rows: List[Dict[str, Any]] = []
    for season in ["2023/24", "2024/25", "2025/26"]:
        source_path = resolve_source_path(base_dir, season)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing official source file: {source_path}")

        raw_df = pd.read_csv(source_path, encoding="utf-8-sig")
        required_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC"]
        missing_columns = [column for column in required_columns if column not in raw_df.columns]
        if missing_columns:
            raise ValueError(f"Source file {source_path} is missing required columns: {missing_columns}")

        season_rows = []
        for _, raw_row in raw_df.iterrows():
            match_date = pd.to_datetime(raw_row["Date"], dayfirst=True, errors="coerce")
            if pd.isna(match_date):
                raise ValueError(f"Invalid date in {source_path}: {raw_row['Date']}")
            home_team = normalize_team_name(raw_row["HomeTeam"])
            away_team = normalize_team_name(raw_row["AwayTeam"])
            if home_team == away_team:
                raise ValueError(f"Self-fixture detected in {source_path}: {home_team} vs {away_team}")
            home_goals = int(float(raw_row["FTHG"]))
            away_goals = int(float(raw_row["FTAG"]))
            home_corners = int(float(raw_row["HC"]))
            away_corners = int(float(raw_row["AC"]))
            if home_corners < 0 or away_corners < 0:
                raise ValueError(f"Negative corner count in {source_path}")
            season_rows.append(
                {
                    "date": match_date.strftime("%Y-%m-%d"),
                    "season": season,
                    "competition": "Serie A",
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_corners": home_corners,
                    "away_corners": away_corners,
                    "total_corners": home_corners + away_corners,
                    "source": "football-data-csv",
                    "source_file_name": source_path.name,
                    "source_url": f"https://www.football-data.co.uk/mmz4281/{season.split('/')[0]}{season.split('/')[1]}/I1.csv",
                    "import_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        if len(season_rows) != 380:
            raise ValueError(f"Expected 380 rows for {season}, received {len(season_rows)}")
        rows.extend(season_rows)

    matches_df = pd.DataFrame(rows)
    matches_df = matches_df.sort_values(["season", "date", "home_team", "away_team"]).reset_index(drop=True)
    matches_df["fixture_id"] = range(1, len(matches_df) + 1)
    matches_df["row_hash"] = matches_df.apply(lambda row: compute_row_hash(row), axis=1)
    matches_df = matches_df.sort_values(["season", "date", "fixture_id"]).reset_index(drop=True)
    matches_df["fixture_id"] = range(1, len(matches_df) + 1)
    return matches_df


def resolve_source_path(base_dir: Path, season: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        base_dir / OFFICIAL_SOURCE_FILES[season],
        repo_root / OFFICIAL_SOURCE_FILES[season],
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def create_historical_sqlite(matches_df: pd.DataFrame, db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        DROP TABLE IF EXISTS matches;
        DROP TABLE IF EXISTS match_stats;
        DROP TABLE IF EXISTS teams;
        DROP TABLE IF EXISTS competitions;
        DROP TABLE IF EXISTS sources;

        CREATE TABLE matches (
            fixture_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            import_date TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            date TEXT NOT NULL,
            season TEXT NOT NULL,
            competition TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_goals INTEGER NOT NULL,
            away_goals INTEGER NOT NULL,
            home_corners INTEGER NOT NULL,
            away_corners INTEGER NOT NULL,
            total_corners INTEGER NOT NULL
        );

        CREATE TABLE match_stats (
            fixture_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            import_date TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_corners INTEGER NOT NULL,
            away_corners INTEGER NOT NULL,
            total_corners INTEGER NOT NULL
        );

        CREATE TABLE teams (
            team_name TEXT PRIMARY KEY,
            competition TEXT NOT NULL,
            source TEXT NOT NULL,
            import_date TEXT NOT NULL,
            row_hash TEXT NOT NULL
        );

        CREATE TABLE competitions (
            competition_name TEXT NOT NULL,
            season TEXT NOT NULL,
            source TEXT NOT NULL,
            import_date TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            PRIMARY KEY (competition_name, season)
        );

        CREATE TABLE sources (
            source TEXT PRIMARY KEY,
            import_date TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            status TEXT NOT NULL
        );
        """
    )

    for _, row in matches_df.iterrows():
        conn.execute(
            "INSERT INTO matches (fixture_id, source, import_date, row_hash, date, season, competition, home_team, away_team, home_goals, away_goals, home_corners, away_corners, total_corners) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(row["fixture_id"]),
                row["source"],
                row["import_date"],
                row["row_hash"],
                row["date"],
                row["season"],
                row["competition"],
                row["home_team"],
                row["away_team"],
                int(row["home_goals"]),
                int(row["away_goals"]),
                int(row["home_corners"]),
                int(row["away_corners"]),
                int(row["total_corners"]),
            ),
        )
        conn.execute(
            "INSERT INTO match_stats (fixture_id, source, import_date, row_hash, date, home_team, away_team, home_corners, away_corners, total_corners) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(row["fixture_id"]),
                row["source"],
                row["import_date"],
                row["row_hash"],
                row["date"],
                row["home_team"],
                row["away_team"],
                int(row["home_corners"]),
                int(row["away_corners"]),
                int(row["total_corners"]),
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO teams (team_name, competition, source, import_date, row_hash) VALUES (?, ?, ?, ?, ?)",
            (row["home_team"], row["competition"], row["source"], row["import_date"], row["row_hash"]),
        )
        conn.execute(
            "INSERT OR IGNORE INTO competitions (competition_name, season, source, import_date, row_hash) VALUES (?, ?, ?, ?, ?)",
            (row["competition"], row["season"], row["source"], row["import_date"], row["row_hash"]),
        )
        conn.execute(
            "INSERT OR IGNORE INTO sources (source, import_date, row_hash, provider_name, status) VALUES (?, ?, ?, ?, ?)",
            (row["source"], row["import_date"], row["row_hash"], "football-data-csv", "imported"),
        )
    conn.commit()
    conn.close()


def build_research_dataset(matches_df: pd.DataFrame) -> pd.DataFrame:
    feature_store = FeatureStore()
    feature_rows = feature_store.transform(matches_df)
    research_df = pd.concat([matches_df.reset_index(drop=True), feature_rows.reset_index(drop=True)], axis=1)
    return research_df


def build_validation_report(matches_df: pd.DataFrame) -> str:
    validations = validate_matches(matches_df)
    lines = [
        "# Database Validation Report",
        "",
        f"- Total matches: {validations['total_matches']}",
        f"- Matches per season: {validations['season_counts']}",
        f"- Duplicate fixtures: {validations['duplicate_fixtures']}",
        f"- Duplicate IDs: {validations['duplicate_ids']}",
        f"- Missing corner values: {validations['missing_corner_values']}",
        f"- Chronological ordering issues: {validations['chronological_issues']}",
        "",
        "## Validation summary",
        "- All required integrity checks passed." if validations["passed"] else "- One or more integrity checks failed.",
    ]
    return "\n".join(lines)


def validate_matches(matches_df: pd.DataFrame) -> Dict[str, Any]:
    season_counts = matches_df.groupby("season").size().to_dict()
    duplicate_fixtures = int(matches_df.duplicated(subset=["date", "home_team", "away_team", "season"]).sum())
    duplicate_ids = int(matches_df["fixture_id"].duplicated().sum())
    missing_corner_values = int(matches_df[["home_corners", "away_corners"]].isna().sum().sum())
    dates = pd.to_datetime(matches_df["date"])
    chronological_issues = int((dates.sort_values().reset_index(drop=True) != dates.reset_index(drop=True)).sum())
    passed = (
        season_counts.get("2023/24", 0) == 380
        and season_counts.get("2024/25", 0) == 380
        and season_counts.get("2025/26", 0) == 380
        and len(matches_df) == 1140
        and duplicate_fixtures == 0
        and duplicate_ids == 0
        and missing_corner_values == 0
        and chronological_issues == 0
    )
    return {
        "total_matches": int(len(matches_df)),
        "season_counts": season_counts,
        "duplicate_fixtures": duplicate_fixtures,
        "duplicate_ids": duplicate_ids,
        "missing_corner_values": missing_corner_values,
        "chronological_issues": chronological_issues,
        "passed": passed,
    }


def build_descriptive_report(matches_df: pd.DataFrame, reports_dir: Path) -> Tuple[str, List[Path]]:
    numeric = matches_df[["home_corners", "away_corners", "total_corners"]]
    summary = numeric.agg(["mean", "median", "std"]).T
    summary["variance"] = numeric.var()
    bins = np.arange(0, 21, 2)
    distribution = pd.cut(matches_df["total_corners"], bins=bins).value_counts().sort_index()
    season_summary = matches_df.groupby("season").agg(mean_total_corners=("total_corners", "mean"), mean_home_corners=("home_corners", "mean"), mean_away_corners=("away_corners", "mean"))

    hist_path = reports_dir / "total_corners_histogram.html"
    qq_path = reports_dir / "total_corners_qq.html"
    corr_path = reports_dir / "corner_correlation_matrix.html"

    fig_hist = px.histogram(matches_df, x="total_corners", nbins=20, title="Total Corners Distribution")
    fig_hist.write_html(hist_path)

    quantiles = np.quantile(matches_df["total_corners"], np.linspace(0, 1, 100))
    theoretical = np.quantile(np.random.normal(size=100), np.linspace(0, 1, 100))
    fig_qq = go.Figure(data=[go.Scatter(x=theoretical, y=quantiles, mode="markers", name="QQ")])
    fig_qq.update_layout(title="QQ Plot of Total Corners", xaxis_title="Theoretical Quantiles", yaxis_title="Observed Quantiles")
    fig_qq.write_html(qq_path)

    corr_fig = px.imshow(matches_df[["home_corners", "away_corners", "total_corners"]].corr(), title="Corner Correlation Matrix")
    corr_fig.write_html(corr_path)

    lines = [
        "# Descriptive Statistics Report",
        "",
        "## Summary",
        summary.to_string(),
        "",
        "## Distribution",
        distribution.to_string(),
        "",
        "## Season comparison",
        season_summary.to_string(),
        "",
        "## Plots",
        f"- Histogram: [total_corners_histogram.html]({hist_path.name})",
        f"- QQ plot: [total_corners_qq.html]({qq_path.name})",
        f"- Correlation matrix: [corner_correlation_matrix.html]({corr_path.name})",
    ]
    return "\n".join(lines), [hist_path, qq_path, corr_path]


def build_data_dictionary(research_df: pd.DataFrame) -> str:
    lines = ["# Data Dictionary", "", "The research dataset contains one row per match with raw match fields and research features generated by the existing feature-store pipeline.", ""]
    for column in research_df.columns:
        dtype = str(research_df[column].dtype)
        description = describe_column(column)
        lines.append(f"## {column}")
        lines.append(f"- Type: {dtype}")
        lines.append(f"- Description: {description}")
        lines.append("")
    return "\n".join(lines)


def describe_column(column: str) -> str:
    descriptions = {
        "fixture_id": "Unique identifier for each fixture.",
        "date": "Match date in ISO format.",
        "season": "Season label.",
        "competition": "Competition label.",
        "home_team": "Home team name.",
        "away_team": "Away team name.",
        "home_goals": "Home goals scored.",
        "away_goals": "Away goals scored.",
        "home_corners": "Home team corner count.",
        "away_corners": "Away team corner count.",
        "total_corners": "Combined corner count.",
        "source": "Data provider source.",
        "import_date": "Import timestamp.",
        "row_hash": "Deterministic hash of the row payload.",
    }
    return descriptions.get(column, "Research feature generated by the existing CornerLab feature pipeline.")


def compute_row_hash(row: Dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_historical_fingerprint(base_dir: Path) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    payload: Dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "source_files": gather_source_file_manifest(base_dir),
        "builder_modules": [],
        "model_artifacts": [],
    }

    for module_path in [
        "src/data/historical_foundation.py",
        "src/data/provenance.py",
        "src/engine/feature_store.py",
        "src/data/normalizer.py",
    ]:
        full_path = repo_root / module_path
        if full_path.exists():
            payload["builder_modules"].append(
                {
                    "path": module_path,
                    "sha256": sha256_file(full_path),
                    "size": full_path.stat().st_size,
                }
            )

    model_dir = repo_root / "models" / "research"
    if model_dir.exists():
        for model_path in sorted(model_dir.rglob("*")):
            if model_path.is_file():
                payload["model_artifacts"].append(
                    {
                        "path": model_path.relative_to(repo_root).as_posix(),
                        "sha256": sha256_file(model_path),
                        "size": model_path.stat().st_size,
                    }
                )

    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def gather_source_file_manifest(base_dir: Path) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    for season in ["2023/24", "2024/25", "2025/26"]:
        source_path = resolve_source_path(base_dir, season)
        manifest.append(
            {
                "season": season,
                "path": source_path.as_posix(),
                "sha256": sha256_file(source_path) if source_path.exists() else None,
                "size": source_path.stat().st_size if source_path.exists() else None,
            }
        )
    return manifest


def persist_cached_artifacts(base_dir: Path, cache_dir: Path, artifact_paths: List[Path]) -> None:
    for artifact in artifact_paths:
        if not artifact.exists():
            continue
        cached_path = cache_dir / artifact.relative_to(base_dir).as_posix() if artifact.is_absolute() else cache_dir / artifact.as_posix()
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, cached_path)


def restore_cached_artifacts(base_dir: Path, cache_dir: Path) -> None:
    for artifact in CACHE_ARTIFACTS:
        cached_path = cache_dir / artifact.as_posix()
        target_path = base_dir / artifact.as_posix()
        if cached_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, target_path)


def load_cache_metadata(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_cache_stats(cache_dir: Path) -> Dict[str, Any]:
    stats_path = cache_dir / "stats.json"
    if not stats_path.exists():
        return {"rebuilds": 0, "cache_hits": 0, "last_rebuild_seconds": None, "last_cache_hit_seconds": None, "last_cache_status": None, "last_cache_fingerprint": None}
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"rebuilds": 0, "cache_hits": 0, "last_rebuild_seconds": None, "last_cache_hit_seconds": None, "last_cache_status": None, "last_cache_fingerprint": None}


def write_cache_stats(cache_dir: Path, stats: Dict[str, Any]) -> None:
    stats_path = cache_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def write_research_cache_report(base_dir: Path, stats: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    report_path = base_dir / "reports" / "research_cache.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    previous_runtime = stats.get("last_rebuild_seconds")
    cached_runtime = stats.get("last_cache_hit_seconds")
    total_runs = int(stats.get("rebuilds", 0)) + int(stats.get("cache_hits", 0))
    hit_ratio = round((int(stats.get("cache_hits", 0)) / total_runs * 100.0), 2) if total_runs else 0.0
    performance_improvement = None
    if previous_runtime and cached_runtime:
        performance_improvement = round(((previous_runtime - cached_runtime) / previous_runtime) * 100.0, 2) if previous_runtime > 0 else 0.0

    lines = [
        "# Historical Research Cache",
        "",
        "## Summary",
        f"- Previous runtime: {previous_runtime if previous_runtime is not None else 'n/a'}s",
        f"- Cached runtime: {cached_runtime if cached_runtime is not None else 'n/a'}s",
        f"- Cache hit ratio: {hit_ratio:.2f}%",
        "- Cache invalidation policy: invalidate when the cache version changes or when any source CSV, builder module, or model artifact fingerprint changes.",
        "- Files cached:",
    ]
    for artifact in metadata.get("artifacts", []):
        lines.append(f"  - {artifact}")
    lines.extend(
        [
            "",
            f"- Performance improvement: {performance_improvement if performance_improvement is not None else 'n/a'}%",
            f"- Fingerprint: {metadata.get('fingerprint', 'n/a')}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    build_historical_database(Path.cwd())
    print("Built historical research foundation")
