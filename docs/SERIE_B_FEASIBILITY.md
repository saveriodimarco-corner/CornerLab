# Serie B corner-market feasibility

Audit date: 2026-10-11

## Objective

Determine whether Italian Serie B can be added to CornerLab without changing
the stable Serie A production pipeline.

The required production market is `alternate_totals_corners`.

## Competition availability

The Odds API reports the following competition as active:

- Sport key: `soccer_italy_serie_b`
- Title: `Serie B - Italy`
- Future events found during the audit: 11

Competition schedules are therefore available.

## Corner-market audit

The event-market inventory was sampled without database writes, cache
persistence, or production configuration changes.

| Fixture | Region | Bookmakers | Required market available |
| --- | --- | ---: | --- |
| US Catanzaro 1929 vs Mantova | `us` | 9 | No |
| Padova vs Carrarese | `us` | 9 | No |
| US Catanzaro 1929 vs Mantova | `eu` | 20 | No |
| US Catanzaro 1929 vs Mantova | `uk` | 20 | No |

No market key containing `corner` was returned in any sample.

The available keys covered match result, spreads, goal totals, both teams to
score, and related alternate markets. Keys such as `totals` and
`alternate_totals` refer to goals and cannot replace corner totals.

Each event-market inventory request cost one API credit. Four paid credits
were used by this audit.

## Decision

The Odds API currently provides Serie B fixtures but does not provide the
corner market required by CornerLab in the tested `us`, `eu`, or `uk`
regions.

Serie B must not be enabled in the production collector, prediction pipeline,
Telegram workflow, or real-bet workflow with the current data source.

This is a point-in-time coverage result, not proof that corner markets can
never appear for any Serie B event. However, the repeated negative result
close to kickoff is sufficient to block implementation against the current
provider.

## Isolation requirements

Any future Serie B implementation must:

- leave all Serie A configuration and schedules unchanged;
- use a separate feature flag, disabled by default;
- use separate reports, models, notification state, and paper-trading state;
- begin in collection-only shadow mode;
- prohibit interactive or real-bet Telegram actions until explicit approval;
- fail closed when genuine Serie B corner odds are unavailable.

## Next feasibility gate

Before implementation, evaluate another provider that explicitly supports:

1. Italian Serie B fixtures and stable fixture identifiers;
2. pre-match total-corner lines and decimal odds;
3. bookmaker and timestamp metadata;
4. historical corner odds suitable for model validation;
5. licensing and quota terms compatible with production use.

Only after a provider passes this gate should a Serie B shadow collector be
implemented.
