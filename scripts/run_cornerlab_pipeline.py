from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collector.collector_config import CollectorConfig
from src.collector.collector_repository import CollectorRepository
from src.collector.fixture_collector import FixtureCollector
from src.collector.live_provider_adapter import LiveProviderAdapter
from src.collector.odds_collector import OddsCollector
from src.collector.result_resolver import ResultResolver
from src.engine.prediction_engine import PredictionEngine
from src.research.advanced_features import build_advanced_feature_dataset
from src.research.feature_selection import run_feature_selection
from src.research.model_benchmark import run_model_benchmark
from src.research.confidence_engine import run_confidence_engine


PIPELINE_VERSION = "0.1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(log_path: Path, stage: str, message: str, level: str = "INFO") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = f"{_utc_now()} [{level}] {stage}: {message}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(entry + "\n")
    print(entry)


def _format_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_execution_summary(path: Path, summary: Dict[str, Any], started_at: str, completed_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# Execution Summary",
            "",
            f"- Status: {summary.get('status', 'unknown')}",
            f"- Started: {started_at}",
            f"- Completed: {completed_at}",
            f"- Pipeline version: {summary.get('pipeline_version', PIPELINE_VERSION)}",
            f"- Dry run: {summary.get('dry_run', False)}",
            "",
            "## Stages",
            f"- Collector: {summary.get('collector', {}).get('status', 'unknown')}",
            f"- Research: {summary.get('research', {}).get('status', 'unknown')}",
            f"- Predictions: {summary.get('predictions', {}).get('status', 'unknown')}",
            "",
            "## Notes",
            *[f"- {note}" for note in summary.get("notes", [])],
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def run_pipeline(base_dir: Path | str | None = None, output_dir: Path | str | None = None, dry_run: bool = False) -> Dict[str, Any]:
    base_dir = Path(base_dir) if base_dir is not None else REPO_ROOT
    output_dir = Path(output_dir) if output_dir is not None else base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / "pipeline.log"
    summary: Dict[str, Any] = {
        "status": "ok",
        "dry_run": bool(dry_run),
        "pipeline_version": PIPELINE_VERSION,
        "collector": {"status": "skipped", "writes": 0},
        "research": {"status": "skipped", "artifacts": []},
        "predictions": {"status": "skipped", "rows": 0},
        "notes": [],
        "errors": [],
        "warnings": [],
    }

    config = CollectorConfig(db_path=base_dir / "data" / "collector.sqlite")
    repo = CollectorRepository(config)
    started_at = _utc_now()
    _append_log(log_path, "START", f"Pipeline starting (dry_run={dry_run})")

    if dry_run:
        repo.record_operation_run(
            status="ok",
            pipeline_version=PIPELINE_VERSION,
            prediction_count=0,
            decision_count=0,
            runtime_seconds=0.0,
            errors=[],
            warnings=[],
            started_at=started_at,
            completed_at=_utc_now(),
        )
        summary["notes"].append("Dry run completed without touching external data sources.")
        _append_log(log_path, "RESULT", "Dry run completed successfully")
        _write_execution_summary(reports_dir / "execution_summary.md", summary, started_at, _utc_now())
        _write_json(output_dir / "reports" / "latest_pipeline_run.json", summary)
        return summary

    try:
        fixture_collector = FixtureCollector(config, repo)
        odds_collector = OddsCollector(config, repo)
        result_resolver = ResultResolver(config, repo)
        live_adapter = LiveProviderAdapter(config)

        try:
            fixtures = live_adapter.fetch_fixtures()
        except Exception as exc:
            fixtures = []
            summary["collector"] = {"status": "error", "writes": 0, "error": _format_error(exc)}
            summary["errors"].append(_format_error(exc))
            _append_log(log_path, "ERROR", f"Collector failed: {_format_error(exc)}", "ERROR")

        collector_writes = 0
        for fixture in fixtures:
            saved = fixture_collector.collect_from_provider(fixture)
            if saved:
                collector_writes += 1
            provider_fixture_id = str(fixture.get("provider_fixture_id") or "")
            if provider_fixture_id:
                odds_rows = live_adapter.fetch_odds(provider_fixture_id)
                if odds_rows:
                    for row in odds_rows:
                        payload = {
                            "fixture_id": saved.get("fixture_id", 0),
                            "bookmaker": row.get("bookmaker", "unknown"),
                            "market": row.get("market", "UNKNOWN"),
                            "line": row.get("line", ""),
                            "side": row.get("side", ""),
                            "decimal_odds": row.get("odd"),
                            "snapshot_timestamp": config.now_utc(),
                            "minutes_to_kickoff": 60,
                            "provider": "the-odds-api",
                            "provider_event_id": str(row.get("source_fixture_id") or provider_fixture_id),
                            "raw_response_hash": "live_the_odds_api",
                            "import_timestamp": config.now_utc(),
                        }
                        stored = odds_collector.collect_odds(payload)
                        if stored is not None:
                            collector_writes += 1

        summary["collector"] = {"status": "ok", "writes": collector_writes, "fixtures": len(fixtures)}

        research_dataset = build_advanced_feature_dataset(base_dir=base_dir, output_dir=output_dir)
        run_feature_selection(base_dir=base_dir, output_dir=output_dir)
        run_model_benchmark(base_dir=base_dir, output_dir=output_dir)
        run_confidence_engine(base_dir=base_dir, output_dir=output_dir)

        summary["research"] = {
            "status": "ok",
            "artifacts": [
                str(output_dir / "data" / "research" / "advanced_features.parquet"),
                str(output_dir / "data" / "research" / "selected_features_regression.json"),
                str(output_dir / "data" / "research" / "model_benchmark_results.csv"),
                str(output_dir / "data" / "research" / "confidence_predictions.parquet"),
            ],
            "rows": int(len(research_dataset)),
        }

        engine = PredictionEngine()
        ratings_path = base_dir / "data" / "processed" / "team_ratings.parquet"
        features_path = output_dir / "data" / "research" / "advanced_features.parquet"
        predictions_path = output_dir / "data" / "predictions" / "predictions.parquet"
        if ratings_path.exists() and features_path.exists():
            predictions = engine.build(str(ratings_path), str(features_path), str(predictions_path))
            summary["predictions"] = {"status": "ok", "rows": int(len(predictions)), "path": str(predictions_path)}
        else:
            summary["predictions"] = {"status": "skipped", "rows": 0, "path": str(predictions_path)}
            summary["warnings"].append("Prediction inputs were not available; predictions were skipped.")
    except Exception as exc:
        summary["status"] = "error"
        summary["errors"].append(_format_error(exc))
        _append_log(log_path, "ERROR", f"Pipeline failed: {_format_error(exc)}", "ERROR")

    completed_at = _utc_now()
    if summary["errors"]:
        summary["status"] = "error"
    else:
        summary["status"] = "ok"

    repo.record_operation_run(
        status=summary["status"],
        pipeline_version=PIPELINE_VERSION,
        prediction_count=int(summary.get("predictions", {}).get("rows", 0)),
        decision_count=0,
        runtime_seconds=0.0,
        errors=summary.get("errors", []),
        warnings=summary.get("warnings", []),
        started_at=started_at,
        completed_at=completed_at,
    )
    _append_log(log_path, "RESULT", f"Pipeline completed with status={summary['status']}")
    _write_execution_summary(reports_dir / "execution_summary.md", summary, started_at, completed_at)
    _write_json(output_dir / "reports" / "latest_pipeline_run.json", summary)
    return summary


if __name__ == "__main__":
    run_pipeline()
