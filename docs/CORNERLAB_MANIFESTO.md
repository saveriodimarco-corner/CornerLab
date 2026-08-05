# CornerLab Manifesto

## Mission
CornerLab exists to determine whether a statistically valid Total Corners betting model can be built from high-quality historical odds data without compromising scientific discipline.

## Scientific Principles
- Prefer externally verifiable evidence over optimism.
- Require a real historical odds sample before any model or strategy claim.
- Separate data procurement from prediction research.
- Validate with a chronological train/validation split and out-of-sample tests.

## Architecture Principles
- Keep the data layer independent from research and betting logic.
- Preserve provenance for every imported dataset.
- Treat provider licences as first-class architecture constraints.

## Validation Principles
- A model is not eligible for production until it is tested on a verified historical sample with opening and closing lines.
- Closing-line efficiency is a core validation target, not an afterthought.

## Anti-overfitting Rules
- Do not train on the future.
- Do not claim edge without out-of-sample evidence.
- Avoid over-parameterised models until dataset scale is proven.
- Reject any model that only works on a cherry-picked subset.

## Data Quality Principles
- Require bookmaker identity, timestamps, fixture mapping and market metadata.
- Require both opening and closing prices for the same market.
- Require clear licence permission for internal research and backtesting.

## Model Governance
- Every model must be traceable to a documented dataset and licence.
- Every model report must show sample size, coverage, and validation method.
- Every procurement decision requires a written evidence review.

## Definition of Done
A dataset is ready for research only when it has verified historical depth, market coverage, timestamp quality, and producer licence clearance.

## Research Workflow
1. Procure a signed sample export.
2. Audit data quality and lineage.
3. Run chronological validation.
4. Report statistical validity and model risk.
5. Only then decide whether to proceed to model development.
