# Customer Segmentation and Retention

A reproducible churn-risk modeling and retention-segmentation study on the
IBM Telco Customer Churn dataset (7,043 customers), with frozen
train/validation/test discipline, calibrated probabilities, declared cost
thresholds, and per-segment retention analysis.

## Overview

This repository rebuilds an earlier churn pipeline that selected models
and thresholds on the test set and dropped rows with blank
`TotalCharges`. The new protocol fixes those defects:

- The Telco CSV is fetched by the Kaggle runner from
  `blastchar/telco-customer-churn`, verified by sha256 against the pinned
  digest `88be4b93...61358a`, and never committed.
- Blank `TotalCharges` values are imputed to 0 (count recorded in the
  data manifest) instead of dropping rows.
- A stratified 60/20/20 train/validation/test split (seed 42) is fixed
  before modeling.
- All preprocessing lives in sklearn Pipelines fit on train only.
- Optuna TPE tunes logistic regression, XGBoost, and LightGBM on train
  with 5-fold stratified CV, ROC-AUC objective, 100 trials, seed 42.
- Sigmoid vs isotonic calibration is chosen per model on validation by
  log loss; the model is selected on validation by PR-AUC.
- Two thresholds freeze on validation before the first test read: a
  cost-optimal threshold under declared costs (FN 780, FP 20) and a
  maximum-precision threshold subject to recall >= 0.80. A sensitivity
  table covers FN/FP ratios {5, 10, 20, 39, 80}.
- The frozen plan (with `card_write_epoch`) is persisted before the test
  partition is touched; `manifest.json` records
  `ordering_ok = card_write_epoch < first_test_read_epoch`.
- Test metrics use 2,000 bootstrap resamples (seed 0): ROC-AUC, PR-AUC,
  log loss, Brier, ECE (15 bins), and precision/recall/F1/expected cost
  at both frozen thresholds.
- Segmentation crosses tenure and MonthlyCharges tertiles (cut points
  from train) and reports, per test segment: n, churn rate, mean
  predicted risk, and flagged counts.
- SHAP global importance on test explains the selected model; it is
  explanatory only.

Retention actions are recommendations under the stated cost assumptions,
not measured savings.

## Data

| Item | Value |
| :--- | :--- |
| Dataset | IBM Telco Customer Churn (`blastchar/telco-customer-churn` on Kaggle) |
| Rows | 7,043 |
| Target | `Churn` (binary) |
| Features | 19 pre-treatment fields; `customerID` excluded |
| Source SHA-256 | `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` |

## Method

```text
load + sha256 verify -> impute TotalCharges blanks -> stratified 60/20/20
-> per-learner: train-only pipeline + Optuna TPE (100 trials, 5-fold CV,
   ROC-AUC) -> sigmoid/isotonic calibration chosen on validation by log
   loss -> validation PR-AUC selects model -> freeze thresholds on
   validation -> persist plan -> single test read -> metrics, bootstrap
   CIs, segments, SHAP
```

## Results

Pending: the Kaggle CPU benchmark has not completed yet. Results will be
curated unedited from the kernel output into `results/` after the run
verifies `manifest.git_sha` and `ordering_ok`.

## Findings

Pending benchmark completion.

## Provenance

- Protocol: `churn-benchmark-v1`
- Seed: 42 (split, tuning, calibration); bootstrap seed 0
- Declared costs: FN 780, FP 20 (assumptions, not measured)
- Kaggle kernel: `sreeharshak101/customer-segmentation-retention-benchmark`
  (CPU), smoke kernel `...-smoke`
- Source pinned by git SHA in `manifest.json`

## Reproduction

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

Full benchmark (Kaggle CPU):

```bash
scripts/kaggle_run.sh          # versions src + data datasets, pushes, polls
scripts/kaggle_run.sh --smoke  # reduced-trial diagnostic kernel first
```

Local diagnostic run (requires the CSV locally):

```bash
uv run churn benchmark --data Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --out benchmark --trials 8 --boot 100 --smoke
```
