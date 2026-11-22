# Plan

## Status

Rebuild in progress. The package, tests, CI, and Kaggle runner are in
place; the CPU smoke and full runs have not completed yet.

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
8. [ ] Local checks green
9. [ ] Kaggle CPU smoke run verified
10. [ ] Kaggle CPU full run verified (git_sha + ordering_ok)
11. [ ] Curate results/, render image.png, finalize README/REVIEW

## Open questions

- Whether the pinned `blastchar/telco-customer-churn` CSV sha256 matches
  the historically committed file (the runner verifies before upload).
