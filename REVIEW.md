# Review

## Defects fixed vs the legacy pipeline

- Model and threshold selection used the test set; now selection is
  validation-only and thresholds freeze before the first test read.
- Blank `TotalCharges` rows were dropped; now imputed to 0 with the
  count recorded.
- FN 780 / FP 20 costs were undeclared; now declared assumptions in the
  frozen plan and README.
- No package, tests, or reproducible runner; now a uv package under
  `src/churn` with pytest, ruff, mypy strict, CI, and a sha256-guarded
  Kaggle runner.

## Verification log

- Local checks: pending.
- Kaggle smoke: pending.
- Kaggle full: pending.

## Notes

- The legacy `Dataset/` CSV remains in git history; it is removed from
  HEAD and sha256-pinned for the Kaggle data dataset.
- SHAP values are explanatory only and never feed selection.
