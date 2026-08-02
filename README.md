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
