# Pytest Performance Audit

## Summary
- Full suite status: SLOW, not hung
- Production-code changes required: NO
- Root cause: the suite spends most of its wall-clock time in a few research-heavy tests that build large historical datasets and fit multiple statistical models, especially the historical research pipeline and the model benchmark/explainability tests.

## Run context
- Command used: `python3 -m pytest -vv --durations=25`
- Observed collected tests: 66
- Observed completed tests: 66
- Observed failed tests: 0
- Observed full-suite duration from a completed plain run: about 29 minutes 37 seconds (1777.10s)
- Confirmed isolated runtime for the slowest test: `tests/data/test_historical_research.py::test_historical_research_pipeline` took 1489.12s (24m 49s) in a dedicated run.

## Slowest tests (confirmed from the verbose run)
1. `tests/data/test_historical_research.py::test_historical_research_pipeline` — 1455.42s in the full verbose run; it builds a full historical SQLite foundation and parquet dataset and is the dominant contributor to the full-suite wall time.
2. `tests/research/test_confidence_engine.py::test_confidence_scores_and_decisions_are_deterministic_and_bounded` — 64.53s in the full verbose run.
3. `tests/research/test_confidence_engine.py::test_confidence_reports_use_actual_decision_state_coverage` — 32.21s in the full verbose run.
4. `tests/research/test_confidence_engine.py::test_confidence_policy_coverage_splits_the_validation_set` — 32.17s in the full verbose run.
5. `tests/research/test_confidence_engine.py::test_confidence_policy_selects_a_subset_that_outperforms_full_predictions` — 32.16s in the full verbose run.
6. `tests/research/test_confidence_engine.py::test_confidence_policy_meets_coverage_and_selective_metrics_are_valid` — 32.12s in the full verbose run.
7. `tests/research/test_confidence_engine.py::test_policy_search_generates_reports_and_targets_accept_coverage` — 32.12s in the full verbose run.
8. `tests/research/test_confidence_engine.py::test_confidence_engine_builds_validation_predictions_and_outputs` — 32.10s in the full verbose run.
9. `tests/research/test_confidence_engine.py::test_confidence_buckets_and_readiness_are_consistent` — 32.07s in the full verbose run.
10. `tests/research/test_advanced_features.py::test_advanced_features_are_generated_without_leakage` — 14.71s in the full verbose run.

## Duration per test (confirmed from the verbose run)
- `tests/data/test_historical_research.py::test_historical_research_pipeline`: 1455.42s
- `tests/research/test_confidence_engine.py::test_confidence_scores_and_decisions_are_deterministic_and_bounded`: 64.53s
- `tests/research/test_confidence_engine.py::test_confidence_reports_use_actual_decision_state_coverage`: 32.21s
- `tests/research/test_confidence_engine.py::test_confidence_policy_coverage_splits_the_validation_set`: 32.17s
- `tests/research/test_confidence_engine.py::test_confidence_policy_selects_a_subset_that_outperforms_full_predictions`: 32.16s
- `tests/research/test_confidence_engine.py::test_confidence_policy_meets_coverage_and_selective_metrics_are_valid`: 32.12s
- `tests/research/test_confidence_engine.py::test_policy_search_generates_reports_and_targets_accept_coverage`: 32.12s
- `tests/research/test_confidence_engine.py::test_confidence_engine_builds_validation_predictions_and_outputs`: 32.10s
- `tests/research/test_confidence_engine.py::test_confidence_buckets_and_readiness_are_consistent`: 32.07s
- `tests/research/test_advanced_features.py::test_advanced_features_are_generated_without_leakage`: 14.71s

## Bottlenecks identified
### 1) Historical dataset build
- The historical research test calls `scripts.build_research_foundation`.
- That path builds a large SQLite database and a parquet research dataset.
- This is a legitimate data-generation workload rather than a hang.
- Evidence: the test reaches this stage and continues running for a long time without producing a timeout or deadlock signal.

### 2) Repeated confidence-engine evaluation work
- The confidence-engine tests spend large amounts of time evaluating many thresholds and policy combinations.
- That work is legitimate and is centered in the confidence-engine pipeline.
- Evidence: the full verbose run shows several confidence-engine tests in the top 10 slowest durations.

### 3) Repeated model fitting
- `src/research/model_benchmark.py` fits several regression and classification models, including:
  - Poisson regressors
  - Negative binomial regressors
  - Ridge regression
  - HistGradientBoostingRegressor
  - Logistic regression
  - HistGradientBoostingClassifier
- These models are fit repeatedly across multiple targets and then used to generate benchmark reports and plots.
- This is legitimate computation and is the main reason the benchmark/explainability tests are noticeably slower than the rest of the suite.

### 4) Report generation and plotting
- The explainability and validation tests generate several markdown reports and HTML/PNG artifacts.
- This adds overhead but is still normal file I/O and plotting work rather than a stall.

## Network-dependent tests
- No evidence of network-dependent hangs or timeouts in the observed runs.
- The provider-related tests are fast and pass locally.
- The full suite runtime is not being driven by outbound network calls.

## Timeouts / deadlocks / subprocesses
- No timeout failures were observed.
- No subprocess leak or deadlock evidence was found.
- The earlier long-running pytest processes were safely stopped before re-running the diagnostic.

## Why the suite looks slow
The suite is slow because it combines:
- one very heavy data-construction test,
- multiple model-training benchmark tests,
- multiple report-generation tests,
- repeated file I/O and artifact writing.

This is consistent with a legitimately expensive suite rather than an infinite loop or stuck job.

## Recommended remediation
- Keep the current test logic intact for correctness.
- If the goal is faster local runs, the main candidates are:
  - reduce the size of the historical research dataset build in tests,
  - cache the built research foundation between tests,
  - avoid re-running full benchmark/explainability pipelines in multiple tests where the same artifacts can be shared,
  - lower the number of model fits or use smaller training sets for test-time benchmarks.

## Production-code changes required
- NO
