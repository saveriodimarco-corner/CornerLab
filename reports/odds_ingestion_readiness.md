# Odds Ingestion Readiness

This report tracks whether the betting layer can rely on validated external corner odds.

- Sprint: 57 (production activation)
- Prematch provider: The Odds API
- Region: us
- Market key: alternate_totals_corners
- Odds format: decimal

## Production Evidence

- Probe odds removed: 4
- Probe fixtures removed: 1
- Current/upcoming Serie A fixtures examined: 7
- Matched provider events: 5
- Unresolved events (logged as MATCH_UNRESOLVED): 2
- Fixtures with odds: 5
- Bookmakers observed: BetRivers

## Line And Side Coverage

- 8.5 OVER: 5
- 8.5 UNDER: 5
- 9.5 OVER: 5
- 9.5 UNDER: 5
- 10.5 OVER: 5
- 10.5 UNDER: 5
- 11.5 OVER: 5
- 11.5 UNDER: 5

## Persistence And Deduplication

- Total The Odds API snapshots in collector DB: 40
- First activation pass inserted: 40
- Immediate second pass inserted: 0
- Immediate second pass skipped (unresolved/TTL): 2

## Verification Tests

- `python3 -m pytest -q tests/data/test_the_odds_api.py -k "fetch_event_odds_includes_additional_market_param or prepared_request_includes_api_key"`
- `python3 -m pytest -q tests/collector/test_live_provider_adapter.py`
- `python3 -m pytest -q tests/test_data_factory_mvp.py -k "collect_odds_uses_source_fixture_id_when_present or snapshot_deduplication or test_first_empty_odds_response_marks_pending_retry"`
