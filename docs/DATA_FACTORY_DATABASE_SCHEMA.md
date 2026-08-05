# CornerLab Data Factory Database Schema

## Tables
- collector_fixtures
- collector_teams
- collector_bookmakers
- collector_markets
- collector_odds_snapshots
- collector_results
- collector_runs
- collector_errors
- collector_provider_usage

## Core fields
- collector_fixtures: fixture_id, provider_fixture_id, competition, season, kickoff_utc, home_team, away_team, status, provider, created_at, updated_at
- collector_odds_snapshots: fixture_id, bookmaker, market, line, side, decimal_odds, snapshot_timestamp, minutes_to_kickoff, provider, provider_event_id, raw_response_hash, import_timestamp
- collector_results: fixture_id, home_score, away_score, home_corners, away_corners, total_corners, settled_at, provider
