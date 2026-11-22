# Plan

## Status

Complete. The Kaggle CPU benchmark ran on `b3c72b5` with
`ordering_ok=true`; results are curated in `results/` and `image.png` is
rendered from them.

## Stack

- Python 3.12-3.13, uv-managed, src layout (`src/churn`)
- pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, shap, typer
- ruff, mypy strict, pytest + pytest-cov
- CI matrix: Python 3.12/3.13 x ubuntu/macos, `actions/checkout@v5`
- Kaggle CPU kernel via `scripts/kaggle_run.sh` (src + data datasets)

## Steps

1. [x] Package skeleton, data loader with sha256 verify, imputation
2. [x] Train-only preprocessing pipeline
3. [x] Learners: majority baseline, logistic, XGBoost, LightGBM with
       Optuna TPE and explicit sigmoid/isotonic calibration
4. [x] Validation-only thresholds + sensitivity table
5. [x] Metrics, ECE, bootstrap CIs, segmentation, SHAP
6. [x] Benchmark orchestration with frozen plan + ordering guard
7. [x] Tests, CI, AGENTS.md, Kaggle smoke/full kernels
8. [x] Local checks green (17 tests, ruff, mypy strict)
9. [x] Kaggle CPU smoke run verified (git_sha b3c72b5, ordering_ok)
10. [x] Kaggle CPU full run verified (git_sha b3c72b5, ordering_ok,
        100 trials, 2000 bootstrap)
11. [x] Curate results/, render image.png, finalize README/REVIEW

## Open questions

- None. The `blastchar/telco-customer-churn` CSV sha256 matched the
  historically committed file (`88be4b93...61358a`); 11 blank
  TotalCharges rows were imputed to 0 (all tenure=0).
