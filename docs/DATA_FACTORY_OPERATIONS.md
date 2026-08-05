# CornerLab Data Factory Operations

## Execution modes
- DRY RUN: no writes, only planning output.
- ONE SHOT: execute one collection cycle.
- SCHEDULED: repeat on the configured interval.

## Operational notes
- Secrets are loaded from the repository-root .env only.
- The collector is resilient to provider failures and records them in the SQLite error log.
- Synthetic odds are not generated.
