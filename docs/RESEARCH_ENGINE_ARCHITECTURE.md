# Research Engine Architecture

## Objective

The Research Engine is the scientific feature layer for CornerLab. Its role is to transform raw football, odds, and context data into reproducible, versioned, provenance-aware feature sets that can later be consumed by downstream analytical and decision layers.

This engine is intentionally independent from:
- Prediction Engine
- Betting Engine
- Kelly Engine
- Confidence Engine
- Feature Engine (if present as a separate runtime layer)

Its responsibility is strictly scientific feature generation and feature lifecycle management.

---

## 1. Responsibilities

The Research Engine owns the following responsibilities:

1. Ingest and normalize raw football and market data from the existing Data Factory and Collector layers.
2. Build scientific features from raw inputs without making decisions or predictions.
3. Maintain a clean feature store with explicit provenance and versioning.
4. Generate feature snapshots for reproducible backtesting and analysis.
5. Support incremental updates, reprocessing, and historical rebuilds.
6. Track lineage from source rows to feature values.
7. Provide a query-friendly feature layer for future downstream consumers.

The Research Engine does not:
- decide bets
- score predictions
- optimize bankroll
- calibrate confidence
- define trading policy

---

## 2. Module Boundaries

The architecture is split into clearly isolated modules.

### 2.1 Data Intake Layer
Responsibilities:
- Read from existing SQLite-backed storage and raw provider artifacts.
- Normalize entities such as fixtures, teams, competitions, bookmaker odds, and market snapshots.
- Standardize timestamps and identifiers.

### 2.2 Research Domain Layer
Responsibilities:
- Define feature semantics and mathematical formulas.
- Implement feature calculations in a deterministic way.
- Separate feature logic from storage and orchestration.

### 2.3 Feature Computation Layer
Responsibilities:
- Execute feature calculations over a defined time window.
- Support per-match, per-team, per-bookmaker, and per-market feature generation.
- Produce atomic feature rows and aggregated feature snapshots.

### 2.4 Feature Store Layer
Responsibilities:
- Persist feature values in SQLite tables.
- Support historical versions and reprocessing.
- Store metadata for provenance, lineage, and recomputation status.

### 2.5 Reproducibility Layer
Responsibilities:
- Track configuration, code version, input snapshot, and execution context.
- Ensure a feature dataset can be reproduced exactly.

### 2.6 Orchestration Layer
Responsibilities:
- Trigger feature builds in the right order.
- Manage incremental refreshes and full rebuilds.
- Mark feature jobs as pending, running, completed, or failed.

### 2.7 Observability Layer
Responsibilities:
- Track feature generation counts, runtime, failures, and quality metrics.
- Surface warnings for missing data or unstable calculations.

---

## 3. Database Layout

The Research Engine must use a dedicated schema to avoid coupling with collector tables. Existing collector tables may be read, but not reused as the primary feature storage layer.

### 3.1 Core Principles
- Separate operational tables from analytical tables.
- Use explicit versioned snapshots rather than overwriting feature history unnecessarily.
- Keep raw provenance references explicit.
- Support incremental updates and full rebuilds.

### 3.2 Proposed Tables

#### research_runs
Tracks execution jobs.

Columns:
- run_id (PK)
- run_name
- run_type (FULL_REBUILD, INCREMENTAL, REPROCESS)
- status
- started_at
- completed_at
- source_snapshot_id
- config_hash
- code_version
- created_at

#### research_snapshots
Defines immutable input snapshots used by a run.

Columns:
- snapshot_id (PK)
- snapshot_label
- created_at
- source_scope
- data_fingerprint
- notes

#### research_feature_sets
Represents a named collection of features.

Columns:
- feature_set_id (PK)
- feature_set_name
- description
- version
- created_at
- is_active

#### research_feature_definitions
Defines all feature logic metadata.

Columns:
- feature_id (PK)
- feature_name
- category
- priority
- description
- mathematical_definition
- formula_version
- update_frequency
- computational_cost
- is_active
- created_at

#### research_match_features
Stores per-match feature values.

Columns:
- feature_value_id (PK)
- feature_set_id
- snapshot_id
- match_id
- fixture_id
- feature_name
- feature_value
- feature_value_type
- computed_at
- provenance_id

#### research_team_features
Stores per-team feature values.

Columns:
- feature_value_id (PK)
- feature_set_id
- snapshot_id
- team_id
- fixture_id
- feature_name
- feature_value
- feature_value_type
- computed_at
- provenance_id

#### research_bookmaker_features
Stores bookmaker or market-specific feature values.

Columns:
- feature_value_id (PK)
- feature_set_id
- snapshot_id
- bookmaker_id
- market_id
- fixture_id
- feature_name
- feature_value
- feature_value_type
- computed_at
- provenance_id

#### research_provenance
Tracks lineage for every computed feature value.

Columns:
- provenance_id (PK)
- source_table
- source_row_id
- source_hash
- transformation_name
- transformation_version
- input_snapshot_id
- created_at

#### research_feature_quality
Stores derived quality indicators for features.

Columns:
- quality_id (PK)
- feature_set_id
- feature_name
- completeness_ratio
- missing_ratio
- outlier_ratio
- stability_score
- computed_at

#### research_feature_dependencies
Tracks feature dependencies.

Columns:
- dependency_id (PK)
- feature_id
- dependency_feature_id
- dependency_type
- created_at

#### research_feature_versions
Tracks version changes for each feature.

Columns:
- feature_version_id (PK)
- feature_id
- version
- changed_at
- change_reason
- formula_hash

---

## 4. Research Pipeline

The research pipeline is a staged process that turns raw data into feature snapshots.

### Stage 1: Domain Normalization
- Read raw fixture, team, odds, and market rows.
- Map providers to canonical entities.
- Standardize time zones and team naming.

### Stage 2: Context Assembly
- Build match context windows.
- Resolve home/away, competition, season, kickoff, and status.
- Attach sportsbook market context where available.

### Stage 3: Feature Computation
- Compute features per category.
- Use rolling windows, aggregate statistics, and relative comparisons.
- Persist outputs and associated provenance metadata.

### Stage 4: Feature Consolidation
- Group computed values into feature sets.
- Validate completeness, missingness, and ranges.
- Store feature snapshots per run.

### Stage 5: Quality Review
- Flag low-confidence or sparse features.
- Produce diagnostics for feature health.

### Stage 6: Release
- Publish a new feature set version.
- Mark it active for downstream consumers.

---

## 5. Feature Pipeline

The feature pipeline is designed around deterministic calculation and composability.

### 5.1 Feature Inputs
Each feature consumes one or more of the following:
- fixture-level raw fields
- team-level historical performance statistics
- match-level context
- bookmaker odds snapshots
- market-level odds snapshots
- external metadata such as rest days or travel distance

### 5.2 Feature Output Contract
Every feature must emit:
- feature name
- feature value
- feature type
- unit or scale
- timestamp
- provenance reference
- version reference

### 5.3 Feature Categories
The following categories define the feature taxonomy.

---

## 6. Statistical Layer

The statistical layer is responsible for the mathematical design of features and the quality of their estimation.

### 6.1 Statistical Principles
- Use robust and interpretable statistics first.
- Prefer ratio-based and rolling-window features over brittle one-off heuristics.
- Separate raw counts from normalized rates.
- Provide uncertainty diagnostics where relevant.

### 6.2 Statistical Utilities
- rolling aggregations
- z-score normalization
- min-max scaling
- percentile ranking
- moving averages
- weighted averages
- season-adjusted rates
- home/away relative measures

### 6.3 Feature Quality Metrics
Each feature should produce:
- completeness ratio
- missingness ratio
- variance
- stability over time
- distribution skewness
- outlier ratio

---

## 7. Caching Strategy

Caching must balance speed and reproducibility.

### 7.1 Cache Types
- raw input cache for normalized source rows
- intermediate aggregate cache for rolling statistics
- feature snapshot cache for released feature sets

### 7.2 Rules
- Cache only deterministic transformations.
- Cache keys must include input snapshot, feature version, and configuration hash.
- Avoid caching non-deterministic or live-updating data without an explicit version.

### 7.3 Invalidation
- invalidated on source snapshot change
- invalidated when feature version changes
- invalidated when calculation configuration changes

---

## 8. Reproducibility

Reproducibility is mandatory.

### 8.1 Requirements
- Every feature run must be tied to a snapshot and configuration hash.
- Each feature must have a versioned formula definition.
- Input rows used by each computation must be traceable.
- The full pipeline must be rerunnable from the same artifact set.

### 8.2 Reproducibility Artifacts
- run manifest
- feature set manifest
- source snapshot fingerprint
- feature definition version list
- calculation configuration hash

---

## 9. Provenance

Provenance is a first-class requirement.

Every feature value must be traceable to:
- the exact source rows used
- the transformation applied
- the version of the transformation logic
- the run that generated it

Provenance should support:
- auditability
- debugging
- root-cause analysis
- reprocessing

---

## 10. Versioning

Versioning must cover data, logic, and output sets.

### 10.1 Versioning Dimensions
- data snapshot version
- feature definition version
- feature set version
- pipeline version
- code revision

### 10.2 Versioning Rules
- A feature change requires a new feature version and a new feature set version if the public contract changes.
- A data snapshot change should trigger a new run and a new feature snapshot.
- Backward-compatible reprocessing should preserve old versions while adding new ones.

---

## Feature Catalog

The feature catalog is grouped by category. Each feature includes a compact definition for design purposes.

### CATEGORY: TEAM STRENGTH
Priority: HIGH

#### 1. team_attack_strength
- Description: Expected attacking output of a team relative to league context.
- Mathematical definition: Weighted average of goals/conceded and corner-producing actions over recent matches, normalized by league baseline.
- Required raw fields: team_id, match_id, home_score, away_score, home_corners, away_corners, competition_id, season
- Update frequency: After each completed match
- Computational cost: Medium
- Dependencies: team_match_history, league_context

#### 2. team_defence_strength
- Description: Expected defensive resistance of a team.
- Mathematical definition: Weighted average of goals conceded and corners conceded adjusted for opponent strength.
- Required raw fields: team_id, match_id, home_score, away_score, home_corners, away_corners
- Update frequency: After each completed match
- Computational cost: Medium
- Dependencies: team_match_history, opponent_strength

#### 3. team_overall_strength
- Description: Combined attacking and defensive balance.
- Mathematical definition: Composite score from attack and defence strength, optionally centered by league average.
- Required raw fields: team_attack_strength, team_defence_strength
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: team_attack_strength, team_defence_strength

### CATEGORY: MATCH TEMPO
Priority: MEDIUM

#### 4. match_tempo_score
- Description: Measure of how fast or open a match typically tends to be.
- Mathematical definition: Average total corners or goals per 90 minutes over recent matches, adjusted by competition context.
- Required raw fields: home_score, away_score, home_corners, away_corners, match_minutes
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: team_match_history, fixture_context

#### 5. average_corners_per_match
- Description: Typical corner count for a team or matchup.
- Mathematical definition: Rolling average of total corners in recent matches.
- Required raw fields: home_corners, away_corners
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: recent_match_window

### CATEGORY: HOME ADVANTAGE
Priority: HIGH

#### 6. home_advantage_corners
- Description: Home-side corner edge relative to away-side context.
- Mathematical definition: Difference between home and away corner averages, adjusted by league baseline.
- Required raw fields: home_corners, away_corners, competition_id, season
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: team_context, league_baseline

#### 7. home_advantage_goal_context
- Description: Home-side attacking edge in relation to match flow.
- Mathematical definition: Relative home goal scoring rate over recent seasons.
- Required raw fields: home_score, away_score, home_team, away_team
- Update frequency: Periodic rebuild
- Computational cost: Low
- Dependencies: historical_fixture_results

### CATEGORY: ATTACK
Priority: HIGH

#### 8. attack_rate
- Description: Rate at which a team generates attacking opportunities that translate to corners.
- Mathematical definition: Rolling ratio of corners created to total attacking actions, normalized over league context.
- Required raw fields: corners_for, shots, possession, xg (if available)
- Update frequency: After each completed match
- Computational cost: Medium
- Dependencies: team_match_history, event_data_if_available

#### 9. attack_pressure_index
- Description: Pressure applied by a team in the attacking third.
- Mathematical definition: Weighted composite of possession share, shots, and attacking transitions.
- Required raw fields: possession, shots, passes, successful_passes
- Update frequency: After each completed match
- Computational cost: Medium
- Dependencies: team_match_history

### CATEGORY: DEFENCE
Priority: HIGH

#### 10. defence_concession_rate
- Description: Rate at which a team concedes dangerous attacking output.
- Mathematical definition: Rolling ratio of corners conceded to opponent attack pressure indicators.
- Required raw fields: corners_against, shots_against, possession_against
- Update frequency: After each completed match
- Computational cost: Medium
- Dependencies: opponent_context

#### 11. defensive_compactness
- Description: Team compactness in defensive shape.
- Mathematical definition: Inverse of conceded attacking pressure, using rolling context windows.
- Required raw fields: conceded_corners, conceded_shots, possession_against
- Update frequency: After each completed match
- Computational cost: Medium
- Dependencies: team_match_history

### CATEGORY: RECENT FORM
Priority: HIGH

#### 12. recent_corner_form
- Description: Short-term corner-producing trend.
- Mathematical definition: Weighted moving average of corners over the last $n$ matches.
- Required raw fields: home_corners, away_corners
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: recent_match_window

#### 13. recent_scoring_form
- Description: Short-term scoring trend.
- Mathematical definition: Weighted moving average of goals scored and conceded.
- Required raw fields: home_score, away_score
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: recent_match_window

### CATEGORY: REST
Priority: MEDIUM

#### 14. rest_days
- Description: Number of days since the previous fixture.
- Mathematical definition: Difference between current fixture kickoff and previous fixture end time.
- Required raw fields: kickoff_utc, previous_fixture_kickoff, previous_fixture_end_time
- Update frequency: For each upcoming fixture
- Computational cost: Low
- Dependencies: fixture_calendar

#### 15. rest_gap_score
- Description: Relative freshness advantage.
- Mathematical definition: Normalized rest-day difference between teams.
- Required raw fields: rest_days_home, rest_days_away
- Update frequency: For each upcoming fixture
- Computational cost: Low
- Dependencies: rest_days

### CATEGORY: TRAVEL
Priority: MEDIUM

#### 16. travel_distance_score
- Description: Travel burden for away team.
- Mathematical definition: Estimated travel distance between previous fixture location and current fixture location.
- Required raw fields: team_location_history, fixture_location
- Update frequency: For each upcoming fixture
- Computational cost: Medium
- Dependencies: travel_context

#### 17. travel_recovery_penalty
- Description: Additional fatigue penalty for long travel and short recovery windows.
- Mathematical definition: Composite of travel distance and rest gap.
- Required raw fields: travel_distance_score, rest_days
- Update frequency: For each upcoming fixture
- Computational cost: Low
- Dependencies: travel_distance_score, rest_days

### CATEGORY: HEAD TO HEAD
Priority: MEDIUM

#### 18. h2h_corner_bias
- Description: Historical corner bias between two teams.
- Mathematical definition: Rolling average of corner differences across past meetings.
- Required raw fields: home_corners, away_corners, home_team, away_team
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: historical_fixture_results

#### 19. h2h_goal_bias
- Description: Historical scoring bias between two teams.
- Mathematical definition: Average goal difference across prior meetings.
- Required raw fields: home_score, away_score, home_team, away_team
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: historical_fixture_results

### CATEGORY: DISCIPLINE
Priority: LOW

#### 20. fouls_per_match
- Description: Average disciplinary pressure.
- Mathematical definition: Rolling average of fouls and cards.
- Required raw fields: fouls, yellow_cards, red_cards
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: match_event_data

#### 21. discipline_pressure_index
- Description: Composite of fouls and cards to estimate match disruption.
- Mathematical definition: Weighted sum of disciplinary events normalized by match context.
- Required raw fields: fouls, yellow_cards, red_cards
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: fouls_per_match

### CATEGORY: CORNERS
Priority: CRITICAL

#### 22. corner_creation_rate
- Description: Team ability to create corners.
- Mathematical definition: Corners-for per match relative to team attack strength and opponent context.
- Required raw fields: home_corners, away_corners, fixture_id, team_id
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: team_attack_strength, recent_corner_form

#### 23. corner_concession_rate
- Description: Team tendency to concede corners.
- Mathematical definition: Corners-against per match relative to opponent strength.
- Required raw fields: home_corners, away_corners, fixture_id, team_id
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: team_defence_strength, recent_corner_form

#### 24. corner_diff_pressure
- Description: Matching pressure in corner differential terms.
- Mathematical definition: Difference between corner creation rate and corner concession rate.
- Required raw fields: corner_creation_rate, corner_concession_rate
- Update frequency: After each completed match
- Computational cost: Low
- Dependencies: corner_creation_rate, corner_concession_rate

### CATEGORY: BOOKMAKER
Priority: HIGH

#### 25. bookmaker_margin_feature
- Description: Market implied margin for a bookmaker.
- Mathematical definition: Sum of inverse implied probabilities minus one, computed from displayed odds.
- Required raw fields: decimal_odds, bookmaker, market, line, side
- Update frequency: Per odds snapshot
- Computational cost: Low
- Dependencies: odds_snapshots

#### 26. odds_movement_feature
- Description: Change in odds between opening and closing snapshots.
- Mathematical definition: Relative movement of decimal odds over time.
- Required raw fields: opening_odds, closing_odds, snapshot_timestamp
- Update frequency: Per odds snapshot update
- Computational cost: Low
- Dependencies: odds_snapshots

### CATEGORY: MARKET
Priority: HIGH

#### 27. market_liquidity_proxy
- Description: Market depth proxy from available bookmaker coverage.
- Mathematical definition: Number of bookmakers offering a market and the variance of quoted odds.
- Required raw fields: bookmaker, market, line, side, decimal_odds
- Update frequency: Per odds update window
- Computational cost: Medium
- Dependencies: bookmaker_features, odds_snapshots

#### 28. market_consensus_feature
- Description: Cross-bookmaker consensus on the same market.
- Mathematical definition: Weighted average of implied probabilities across bookmakers.
- Required raw fields: bookmaker, market, line, side, decimal_odds
- Update frequency: Per odds update window
- Computational cost: Medium
- Dependencies: odds_snapshots

---

## Implementation Roadmap

### Sprint 20 — Feature Contract and Schema Foundation
Deliverables:
- finalize feature definition registry
- finalize research schema tables
- define feature metadata contract
- define provenance schema
- define run manifest format

Testability:
- A feature can be registered and stored without computation.

### Sprint 21 — Baseline Feature Generation
Deliverables:
- implement team strength, recent form, home advantage, and basic corner proxy features
- build first feature set
- support backfill on historical data

Testability:
- A historical feature set can be generated and validated against known examples.

### Sprint 22 — Odds and Market Feature Layer
Deliverables:
- implement bookmaker margin, odds movement, market liquidity, and consensus features
- connect to collector odds snapshots
- support market-specific feature generation

Testability:
- Market features are reproducibly generated from stored odds snapshots.

### Sprint 23 — Incremental Processing and Provenance
Deliverables:
- incremental feature refresh pipeline
- provenance lineage persistence
- snapshot-based caching
- feature quality diagnostics

Testability:
- Reprocessing a dataset produces the same feature values for the same snapshot.

### Sprint 24 — Feature Release and Validation
Deliverables:
- feature set publication workflow
- active feature set versioning
- validation reports and drift diagnostics
- handoff contracts for downstream engines

Testability:
- An active feature set can be exported and consumed by a downstream consumer without ambiguity.

---

## Summary

The Research Engine should be a disciplined, versioned, provenance-aware scientific feature system. It must remain entirely separate from decision-making, betting, and confidence logic. Its core purpose is to produce stable, explainable, and reproducible feature values from raw football and odds data.
