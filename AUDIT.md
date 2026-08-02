# CornerLab Technical Audit

## Executive Summary

The current project is structurally coherent for a Sprint-based data engineering prototype. The engine modules are separated by concern, the test suite is passing, and the core pipeline from raw match data to ratings, features, and predictions is functional.

However, the implementation still shows several architectural and methodological weaknesses that should be addressed before treating the system as production-grade.

## Verification Snapshot

- Test suite status: 10 tests passed via `python3 -m pytest -q`
- Coverage: not measured in the current environment because coverage reporting was not configured and the coverage plugin was not available
- Runtime artifacts: ratings, features, and predictions parquet outputs are generated successfully

---

## Critical Issues

1. Prediction logic is not fully aligned with the requested weighted-rating contract
   - The prediction engine consumes feature-derived values rather than directly using the weighted team rating outputs as its primary source of truth.
   - This weakens traceability and makes it harder to reason about how ratings affect predictions.

2. Feature generation is computationally expensive for larger datasets
   - The feature store recomputes rating state incrementally for each match while iterating over the full prior history.
   - This creates a scalability bottleneck and will degrade quickly as match volume grows.

---

## High Priority

1. Architecture is still too tightly coupled between stages
   - The feature store depends directly on the team rating engine and uses a custom feature construction flow.
   - The prediction engine depends on the feature-store output shape rather than a shared, explicit interface contract.
   - This increases the cost of changing one stage without touching the others.

2. Data validation is incomplete for real-world production inputs
   - Required columns are checked, but there is no strong validation around:
     - duplicate matches
     - invalid team names
     - inconsistent date ordering
     - unexpected value ranges
     - empty seasons or malformed strings

3. Error handling is inconsistent
   - Some methods raise clear `ValueError`s, but others return default values silently when data is missing.
   - This can hide data quality issues and make debugging more difficult.

4. Statistical assumptions in the Poisson model are simplistic
   - The current prediction engine uses a deterministic Poisson-based approach, but it does not account for overdispersion, zero inflation, or varying home/away dispersion.
   - This is acceptable for a baseline framework, but it is not yet robust enough for production confidence.

---

## Medium Priority

1. Code duplication exists in data loading and validation logic
   - The engine modules each implement similar file-loading and column-checking behaviors.
   - This duplicates validation and makes future changes error-prone.

2. SOLID principles are only partially followed
   - The current classes are focused and readable, but they mix data loading, transformation, validation, and persistence responsibilities.
   - This makes the modules harder to extend and test independently.

3. Missing or weak configuration management
   - Key parameters such as the EWMA alpha, convergence threshold, and prediction thresholds are hard-coded in class constructors and methods.
   - There is no central configuration layer or environment-driven settings model.

4. Logging quality is limited
   - Logging exists in the Streamlit app entrypoint, but the engine modules do not emit actionable operational logs.
   - There is no structured logging, no error-level context, and no per-stage audit trail.

5. Type hints are present but not exhaustive
   - Most public methods are typed, but the code uses broad `Any` and `Dict` shapes in a few places.
   - This makes the interfaces less expressive and can reduce maintainability as the project scales.

6. Some dead or low-value code paths remain
   - Several helper methods and intermediate variables appear to exist primarily to support the current implementation approach rather than a clear abstraction.
   - These should be cleaned up or moved behind clearer interfaces.

---

## Low Priority

1. Dependency quality could be improved
   - The project includes several dependencies that are not used directly in the current implementation path, including some that appear more relevant to future experimentation than the current baseline.
   - This makes dependency maintenance more costly than necessary.

2. Security posture is minimal but acceptable for a local prototype
   - No hard-coded secrets or sensitive credentials were found.
   - However, the project currently lacks explicit input sanitization, trusted-path controls, and safer file handling for future production deployments.

3. Test coverage is good at the unit level but not yet comprehensive
   - The current suite covers the core engines and their main output contracts.
   - It does not yet cover edge cases such as malformed dates, empty datasets, duplicate records, or large-scale performance behavior.

---

## Technical Debt

- The pipeline is still a prototype pipeline rather than a stable domain model.
- The engine modules are functional but not yet fully abstracted behind a shared data contract.
- There is no centralized validation layer or configuration registry.
- The project lacks an explicit data quality framework and a stronger experiment/benchmarking layer.
- Feature generation and prediction logic are coupled to the current data shape and should be made more interface-driven.

---

## Improvement Roadmap

### Phase 1: Stabilize the foundation
- Introduce a shared validation layer for match input data.
- Centralize configuration for alpha, thresholds, and output paths.
- Add structured logging around each engine stage.

### Phase 2: Improve architecture
- Refactor the engine modules around a clearer interface contract.
- Move common file-loading and schema-validation logic into a shared utility layer.
- Reduce coupling between the teams-rating, feature-store, and prediction modules.

### Phase 3: Improve production readiness
- Add stronger tests for malformed and edge-case data.
- Introduce explicit data quality checks and anomaly handling.
- Re-evaluate the Poisson model and consider richer statistical formulations for corner totals.

### Phase 4: Performance and scalability
- Optimize feature generation to avoid repeated recomputation over prior rows.
- Add benchmark tests for larger datasets.
- Consider vectorized or batch-oriented transforms for long-run usage.
