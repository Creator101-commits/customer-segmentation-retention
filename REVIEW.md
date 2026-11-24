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

- Local checks: 17 pytest pass (90% coverage), `ruff check .` clean,
  `mypy src` clean (14 files, strict), U+2014 scan clean.
- Kaggle smoke (`...-smoke`, kernel v1, ~90s compute): COMPLETE on
  `b3c72b5`, `ordering_ok=true`, artifacts in
  `artifacts/kaggle/20261121-200655/`.
- Kaggle full (`...-benchmark`, kernel v1, ~275s compute): COMPLETE on
  `b3c72b5`, `ordering_ok=true`, 100 trials / 2000 bootstrap,
  `smoke=false`, artifacts in `artifacts/kaggle/20261121-201615/`.
- Curation: all six `results/` files verified byte-identical (`cmp`)
  against the kernel output; `image.png` rendered from `results/` only
  by `scripts/render_figure.py` (portfolio v2 style, hard layout checks).

## Residual notes

- sklearn emits benign "X does not have valid feature names" warnings
  from LightGBM during calibration scoring; predictions are unaffected.
- The cost-optimal threshold (0.0243) flags nearly all customers because
  the declared FN:FP cost ratio is 39:1; this is reported as a property of
  the assumed costs, with the sensitivity table for other ratios.
- SHAP values are explanatory only and never feed selection.

## Notes

- The legacy `Dataset/` CSV remains in git history; it is removed from
  HEAD and sha256-pinned for the Kaggle data dataset.
- SHAP values are explanatory only and never feed selection.
