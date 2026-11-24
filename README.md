# Customer Segmentation and Retention

Which telecom customers are about to leave, and which of them are worth
calling? This project ranks 7,043 IBM Telco customers by churn risk, turns
those scores into two concrete flagging policies, and checks whether the
scores hold up segment by segment. Every choice (model, calibration,
cut-off) is made on validation data, and the test set is read once.

![Churn benchmark: discrimination, frozen thresholds, segment calibration, segment grid and SHAP importance](image.png)

The figure follows the study end to end: (a) how well each model separates
churners, (b) what the two frozen cut-offs catch and cost, (c) whether
predicted risk matches reality inside each segment, (d) where churn
concentrates, (e) which fields drive the predictions.

## Results

| Model | ROC-AUC (test) | PR-AUC (test) | ECE |
| :--- | ---: | ---: | ---: |
| majority baseline | 0.5000 | 0.2654 | 0.265 |
| logistic regression | 0.8421 [0.821, 0.862] | 0.6183 [0.570, 0.670] | 0.023 |
| **xgboost (selected)** | 0.8465 [0.825, 0.867] | 0.6372 [0.589, 0.686] | 0.025 |
| lightgbm | 0.8411 [0.820, 0.862] | 0.6189 [0.570, 0.671] | 0.022 |

- **The three models are a tie.** XGBoost won on validation (PR-AUC
  0.6455) and reaches 0.847 ROC-AUC on test, but its interval overlaps
  logistic regression and LightGBM almost entirely. The honest claim is
  "all three sit near 0.84, far above the 0.500 baseline", not "xgboost is
  best".
- **The scores are trustworthy, not just well ordered.** After isotonic
  calibration, test ECE is 0.022 to 0.025, and observed churn tracks
  predicted risk in all nine segments (0.728 predicted vs 0.679 observed
  in the worst one).
- **Churn concentrates sharply.** New customers on high monthly bills
  churn at 0.679 (n=81); long-tenure customers on low bills churn at
  0.015 (n=135). That spread is what makes segmentation useful.
- **Contract type dominates.** Mean |SHAP| ranks Contract (0.789) ahead of
  tenure (0.324) and MonthlyCharges (0.233), which matches the segment
  pattern.
- **The cost-optimal policy is barely a policy.** Under the declared costs
  a missed churner is worth 39 wasted calls, so the cut-off collapses to
  0.0243 and flags nearly everyone: recall 0.995 at precision 0.310. The
  recall-constrained cut-off (0.2149) is the usable one, with precision
  0.472 at recall 0.845.

| Frozen threshold | Cut-off | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: | ---: |
| cost-optimal (FN 780 / FP 20) | 0.0243 | 0.310 | 0.995 | 0.472 |
| max precision at recall >= 0.80 | 0.2149 | 0.472 | 0.845 | 0.605 |

## How it works

```text
load + sha256 verify -> impute blank TotalCharges -> stratified 60/20/20
-> tune each learner on train (Optuna TPE, 100 trials, 5-fold CV)
-> calibrate and pick the model on validation -> freeze both cut-offs
-> write the plan -> read test once -> metrics, bootstrap CIs, segments, SHAP
```

This is a rebuild of an earlier pipeline that had selected its model and
threshold on the test set and dropped rows with a blank `TotalCharges`.
The ordering above is enforced in code: `manifest.json` records that the
plan was written before the first test read.

## Run it

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

The full benchmark runs on Kaggle CPU, which downloads the dataset and
verifies its checksum:

```bash
scripts/kaggle_run.sh           # full run
scripts/kaggle_run.sh --smoke   # reduced-trial check first
```

## Notes

- Retention costs (FN 780, FP 20) are declared assumptions, so the
  expected costs are policy comparisons, not measured savings.
- SHAP importances explain the model, not causes of churn.
- Data: IBM Telco Customer Churn (`blastchar/telco-customer-churn`), 7,043
  rows, 19 pre-treatment features, pinned by SHA-256 and never committed.
- Protocol `churn-benchmark-v1`, seed 42, bootstrap seed 0. Curated kernel
  output is in `results/`, and `image.png` is rendered from it.
