# CornerLab Data Factory Architecture

## Scope
The data factory MVP collects fixture metadata, results, and odds snapshots into a SQLite-backed repository without touching prediction or betting logic.

## Components
- collector_config.py: environment loading, redaction, and runtime settings.
- collector_repository.py: SQLite schema and persistence helpers.
- fixture_collector.py: fixture creation and updates.
- odds_collector.py: odds snapshot ingestion.
- result_resolver.py: result resolution.
- snapshot_engine.py: snapshot validation and selection helpers.
- scheduler.py: one-shot and dry-run orchestration.
- collector_health.py: health summarization.

## Commands
- python3 scripts/run_data_factory_once.py --dry-run
- python3 scripts/run_data_factory_once.py
- python3 scripts/check_data_factory_health.py
