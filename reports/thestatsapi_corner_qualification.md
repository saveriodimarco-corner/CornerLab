# TheStatsAPI historical corner odds qualification

## Authentication and connectivity

- Authentication: FAIL
- Loaded API key from: .env
- Health endpoint: PASS
- Protected competition and season endpoints returned 403 with the provider's revoked/subscription error.

## Evidence captured

- health endpoint responded 200
- /football/competitions returned 403: API key has no active subscription plan
- /football/competitions/it/serie-a returned 403: API key has no active subscription plan
- /football/competitions/it/serie-a/seasons returned 403: API key has no active subscription plan

## Requested qualification metrics

- Serie A seasons resolved: none
- Competition ID: unavailable
- Season IDs: unavailable
- Total matches indexed per season: unavailable
- Matches marked as having odds: unavailable
- Sampled matches: 0
- Matches with any odds: 0
- Matches with genuine corner odds: 0
- Corner coverage percentage: 0.00%
- Matches with both Over and Under: 0
- Matches with opening odds: 0
- Matches with closing/last-seen odds: 0
- Bookmakers found: none
- Target lines found: none
- Fixture mapping success rate: 0.00%
- API requests consumed: 4
- Final verdict: NOT SUITABLE FOR CORNERLAB

## Notes

The qualification was stopped before any historical match sampling because the provider rejected the key for the protection endpoints required to resolve Serie A seasons and fixtures. The evidence therefore supports a non-purchase decision for this sprint.
