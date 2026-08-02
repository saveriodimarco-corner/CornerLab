# CornerLab

CornerLab is a data-driven corner prediction platform with a modular engine pipeline for ratings, features, and predictions.

## Data Foundation

Sprint 5 introduced a shared data foundation for the project:

- Centralized configuration in [src/config.py](src/config.py)
- Data validation in [src/utils/validator.py](src/utils/validator.py)
- Shared loading support in [src/utils/data_loader.py](src/utils/data_loader.py)
- Cache reuse support in [src/utils/cache.py](src/utils/cache.py)
- Typed data contracts in [src/models/data_contracts.py](src/models/data_contracts.py)

## Project Structure

- src/engine
- src/ui
- src/database
- src/models
- src/utils
- tests
- docs
- data/raw
- data/processed
- data/features
- data/predictions

## Run the app

```bash
pip install -r requirements.txt
streamlit run src/ui/app.py
```

## Run tests

```bash
python3 -m pytest -q
```

## Sprint 6 – Data Acquisition Layer

The repository now includes a multi-source acquisition pipeline that can ingest data from providers, normalize rows, run data-quality checks, persist imported rows to SQLite, and generate a quality report.

### What changed
- Added an abstract provider interface in [src/data/providers/base.py](src/data/providers/base.py)
- Added concrete providers for football data, API football, and CSV input in [src/data/providers](src/data/providers)
- Added normalization for dates, team names, competitions, seasons, and corner counts in [src/data/normalizer.py](src/data/normalizer.py)
- Added quality checks and report generation in [src/data/quality.py](src/data/quality.py)
- Added SQLite persistence for matches, match statistics, teams, competitions, and sources in [src/data/database.py](src/data/database.py)
- Added an import command at [scripts/import_data.py](scripts/import_data.py)

### Import data

```bash
python3 scripts/import_data.py
```
