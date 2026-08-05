# CornerLab Data Factory Recovery

## Recovery approach
- Re-run the one-shot collector to reprocess fixtures idempotently.
- Replay snapshots safely because duplicates are suppressed by a timestamp window.
- Use the error log to inspect provider issues and retry later.

## Commands
- python3 scripts/run_data_factory_once.py --dry-run
- python3 scripts/run_data_factory_once.py
- python3 scripts/check_data_factory_health.py
