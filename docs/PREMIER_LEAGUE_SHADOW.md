# Premier League shadow collection

Initial validation date: 2026-10-30

## Status

The Premier League integration is experimental and collection-only.

It does not run paper trading, send Telegram messages, create betting
confirmations, or write to the Serie A production database.

## Provider feasibility

The Odds API reports `soccer_epl` as active and returned 21 future fixtures
during the initial audit.

For Chelsea vs Manchester United, using the production `us` region:

- `alternate_totals_corners` was available;
- 11 bookmakers exposed event markets;
- 6 bookmakers returned normalized corner odds;
- lines 8.5, 9.5, 10.5, and 11.5 were available;
- both `OVER` and `UNDER` outcomes were available.

The first normalized sample contained 46 rows.

## Isolation

The shadow collector uses:

- database: `data/premier_league_shadow/collector.sqlite`;
- report: `reports/premier_league_shadow_latest.json`;
- competition: `Premier League`;
- sport key: `soccer_epl`.

The existing Serie A database remains `data/collector.sqlite` and is not
modified by the shadow job.

The shadow job has no systemd timer and must be run manually until it passes
the remaining readiness gates.

## API safety

A run performs a free quota check before paid odds requests.

The default limit is one paid request per execution. Fixtures beyond that
limit are recorded with the `paid_request_limit` skip reason and can be
collected by later runs.

If the remaining quota cannot be determined, the job performs no paid odds
requests.

## Duplicate handling

The provider may return more than one price for the same bookmaker, market,
line, and side.

The shadow collector records duplicate and conflicting-duplicate counts and
retains the highest valid decimal price for each combination before writing
a snapshot.

## First shadow run

The first isolated run produced:

- 21 fixtures fetched;
- 1 paid odds request;
- 52 raw odds rows;
- 44 stored snapshots;
- 8 duplicate rows removed;
- 7 conflicting duplicate rows;
- 1 fixture with stored odds.

Paper trading, Telegram, and real betting remained disabled.

## Remaining gates

Before any Premier League decision or notification workflow is enabled:

1. collect multiple matchdays of genuine corner snapshots;
2. measure fixture, bookmaker, line, and time-to-kickoff coverage;
3. verify completed-match corner results and team-name mappings;
4. train and validate competition-specific models;
5. run an isolated Premier League paper-trading phase;
6. review duplicate-price behavior and reference-price selection;
7. obtain explicit approval before enabling Telegram messages;
8. keep real betting disabled until a separate approval.
