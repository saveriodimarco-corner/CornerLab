# Premier League forward validation

Initial feasibility date: 2026-10-30

## Status

The Premier League workflow is active in isolated forward-validation mode.

It collects fixtures and corner odds, produces frozen-model predictions,
sends explicitly labelled Telegram simulations, and settles completed
forward observations and simulated bets.

It does not create real Premier League bets, use the real bankroll, or write
to the Serie A collector database and reports.

## Safety contract

Every forward artifact and runtime report records:

- `mode`: `forward_validation_only`;
- `real_betting_enabled`: `false`;
- `bankroll_real_enabled`: `false`;
- `production_database_writes`: `false`.

Telegram messages are labelled:

`SIMULAZIONE — NESSUNA SCOMMESSA REALE`

Telegram interactions use the isolated directory
`data/telegram_simulation`. Premier League fixture identifiers are prefixed
with `epl:` so they cannot collide with Serie A fixture identifiers.

## Provider feasibility

The Odds API reports `soccer_epl` as active.

The `alternate_totals_corners` market provides normalized `OVER` and `UNDER`
prices for lines 8.5, 9.5, 10.5, and 11.5. The shadow collector uses the
production `us` region and records duplicate and conflicting-duplicate
counts before retaining the highest valid decimal price for each bookmaker,
market, line, and side.

Completed Premier League scores and corners are resolved from the current
Football-Data `E0` CSV. Recent-result cache entries expire after 15 minutes;
historical results retain the 84-hour cache.

## Historical validation

The extended validation dataset contains 3,880 completed Premier League
matches from 2016/17 through the first 80 completed matches of 2026/27.

The frozen forward model is:

- model: `poisson_alpha_10_online_bias`;
- training rows: 3,800;
- online calibration rows: 80;
- calibration cutoff: 2026-10-25;
- eligible lines: 8.5 and 11.5.

Historical and temporal validation improved most MAE and Brier comparisons,
but the complete acceptance protocol did not pass because one out-of-sample
mean Brier comparison remained slightly worse than its baseline.

For that reason the model is authorized only for forward observation and
Telegram simulation. It is not authorized for real betting.

## Runtime isolation

The workflow uses:

- shadow database:
  `data/premier_league_shadow/collector.sqlite`;
- immutable prediction ledger:
  `data/premier_league_shadow/forward_predictions.jsonl`;
- Telegram delivery state:
  `data/premier_league_shadow/forward_telegram_state.json`;
- model and validation cache:
  `data/provider_cache/premier_league_validation/`;
- current scoring reports:
  `reports/premier_league_forward_current.json` and `.csv`;
- settlement report:
  `reports/premier_league_forward_settlement_latest.json`.

The Serie A database remains `data/collector.sqlite`.

## Hourly pipeline

The installed `cornerlab-premier-league-shadow.timer` runs at minute 20 of
every hour, separately from the Serie A jobs at minute 00.

Each run performs, in order:

1. isolated fixture and odds collection;
2. forward settlement for completed fixtures;
3. scoring with the verified frozen artifact;
4. deduplicated Telegram simulation alerts.

The default collection limit is one paid Odds API request per run. If quota
cannot be determined, no paid request is made.

The Odds API client retries only transient network failures, HTTP 429, and
HTTP 5xx responses. Authentication and other non-transient errors fail
immediately.

## Telegram simulation policy

Only fixtures for today or tomorrow in Rome time are considered.

A price-check notification requires:

- an eligible line;
- complete team history;
- data-quality score of 1.0;
- predicted probability of at least 70%;
- a maximum of five notifications per cycle.

Ranking is independent of the provider price. The user must enter the actual
bet365.it price. The bot rejects a price below the existing production
minimum derived from probability, market edge, expected value, and minimum
decimal odds.

Accepted interactions still affect only the simulation bankroll.

## First forward sample

The first forward run produced:

- 21 future fixtures;
- 8 fixtures with eligible odds;
- 32 immutable observations;
- 5 Telegram price checks;
- 0 real Premier League bets.

All five initial prices checked on bet365.it were below their required
minimum and were recorded as `SKIPPED`.

## Settlement

Forward settlement starts no earlier than 2 hours and 30 minutes after
kickoff.

For each completed observation it records:

- final home, away, and total corners;
- win, loss, or void status;
- Brier score;
- unit profit or loss at the originally observed price.

Any confirmed Premier simulation is settled through the isolated simulation
ledger. The real bankroll is never passed to the Premier settlement
function.

## Gates before real activation

Real Premier League betting must remain disabled until all of the following
are complete:

1. accumulate a meaningful forward sample across multiple matchdays;
2. verify fixture, bookmaker, line, and time-to-kickoff coverage;
3. review Brier score and calibration by eligible line;
4. measure forward economic performance using immutable pre-match prices;
5. verify settlement and team mappings across promoted clubs;
6. confirm model stability after the current-season regime shift;
7. complete a separate code and operational review;
8. obtain explicit approval for real Telegram confirmations and bankroll
   access.

## Current-season history and no-play notifications

Each hourly Premier League pipeline refreshes the completed current-season
E0 results before live scoring.

New completed matches are appended incrementally to
`matches_extended.parquet`. Existing fixture identifiers are preserved,
result corrections update the existing row, and repeated runs are
idempotent.

This refresh updates only the historical inputs used to construct live
features. It does not retrain, replace, or modify the frozen forward model.

For each Premier League matchday, CornerLab also sends one informational
Telegram message when:

- no forward candidate satisfies the minimum requirements; or
- every simulated price check for that matchday has been marked as skipped.

The message is sent at most once per Rome calendar date, from six hours
before the first remaining fixture. It contains no betting controls and is
explicitly labelled `NON GIOCARE`.

If any simulated check is still pending, the message waits. If any simulated
bet was placed, the no-play message is not sent.

Both features keep real Premier League betting and the real bankroll
disabled.

## Market-aware readiness result

The accelerated validation used 460 strictly walk-forward Premier League
matches from 2025/26 and 2026/27, with 15,610 historical corner prices
captured 90 minutes before kickoff.

The model did not pass promotion:

- model Brier: 0.22237;
- de-vigged market Brier: 0.21905;
- qualified historical bets: 20;
- historical flat-stake ROI: -14.85%;
- optimal model weight against the market: 0%;
- qualified forward bets after all production gates: 0.

Raw positive edge is not sufficient. A play must also pass probability,
price, EV, quality, and minimum-price gates.

Premier League real betting and the real bankroll remain disabled. Promotion
is never automatic: even after every statistical gate passes, a separate
explicit approval and a small-stake rollout are required.

## Three-season historical market validation

Historical corner-market coverage now contains 840 completed matches:

- 380 matches from 2024/25;
- 380 matches from 2025/26;
- 80 matches from 2026/27.

The dataset contains 28,004 normalized historical odds rows. Walk-forward
predictions use only seasons and observations available before each match.

The combined model Brier score is 0.217236, compared with 0.216529 for the
market. The 43 qualified historical selections produced 1.215 units of
profit and 2.83% ROI, with a 95% ROI interval from -19.69% to 24.95%.

Results are not stable across seasons: 2024/25 returned 18.20%, 2025/26
returned -22.93%, and the first 80 matches of 2026/27 returned 4.00%.

The formal readiness decision remains `NOT_READY`. Real Premier League
betting, automatic promotion, and use of the real bankroll remain disabled.

The historical backfill consumed 8,649 subscription credits and left 11,227
credits available at completion. No further historical API backfill is
currently authorized or required.

## Locked market-blend experiment

A single market-blend weight was selected using only the 2024/25 development
season. The selected blend uses 80% model probability and 20% no-vig market
probability.

On the independent 2025/26 validation season, the blend improved the raw
model Brier score from 0.220639 to 0.219568, but did not beat the market
score of 0.217074. Only one selection passed the decision policy, which is
insufficient for ROI inference.

On the separate 2026/27 observation period, four selections produced
-22.75% ROI. The locked experiment therefore failed its validation gates.

The experiment does not modify the production model and cannot enable real
betting, automatic promotion, or use of the real bankroll.
