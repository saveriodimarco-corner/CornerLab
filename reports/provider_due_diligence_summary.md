# Provider due diligence summary

## Scope

This sprint reviewed public documentation, configured credentials, and authenticated API responses for the listed football-odds providers. No model, confidence, or betting logic was modified.

## Live-tested providers

- API-Football / API-Sports: authenticated and live fixture data were verified.
- TheStatsAPI: health endpoint was authenticated, but protected competition and season endpoints were rejected with a subscription error.

## Evidence summary

- No provider in the current environment produced verified historical Serie A corner odds with opening and closing prices for the requested target lines.
- API-Football returned fixture data but no odds payload for the sampled fixture.
- TheStatsAPI access was blocked by the provider before any historical competitions or seasons could be resolved.

## Recommended outcome

Commercial quotes are required before any provider can be approved for CornerLab historical backtesting. Current evidence is insufficient to support a GO recommendation.
