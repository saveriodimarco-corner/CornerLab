# Provider final recommendation

## Preferred historical provider

No provider can be recommended for historical backtesting on the current evidence. Commercial quotes and licence review are still required.

## Preferred live provider

API-Football / API-Sports is the strongest current live-testing option for fixture and league metadata, but it did not return a usable corner-odds payload for the sampled fixture.

## Recommended architecture

Use a staged architecture: keep the current research stack for internal validation, then add a licensed historical-odds vendor only after a contract and sample export are available. A multi-provider approach is not yet justified without verified corner-market samples.

## Remaining evidence gaps

- Verified historical Serie A corner odds with opening and closing prices.
- Verified target-line coverage for 8.5, 9.5, 10.5, 11.5.
- Licensed internal research and permanent caching rights.
- Resolution of fixture mapping and settlement availability.

## Exact next action

Request a sales call with at least one vendor that offers historical betting odds archives and ask for a sample export of Serie A Total Corners odds spanning at least three completed seasons. Do not proceed to implementation until the sample contains opening and closing odds, bookmaker identity, timestamps, and licence terms suitable for internal backtesting.
