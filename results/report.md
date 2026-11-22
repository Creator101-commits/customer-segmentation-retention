# Churn Benchmark Report

- git SHA: `b3c72b546fb6ad979e46cbc349b9e3c43e3c6dc5`
- Protocol: churn-benchmark-v1
- Selected model (validation PR-AUC): **xgboost** (PR-AUC 0.6455)
- ordering_ok: True

## Test metrics (frozen, 95% bootstrap CI, 2000 resamples)

| Metric | Value |
| :--- | :--- |
| ROC-AUC | 0.8465 [0.8251, 0.8671] |
| PR-AUC | 0.6372 [0.5894, 0.6861] |
| Log loss | 0.4136 [0.3853, 0.4422] |
| Brier | 0.1350 [0.1242, 0.1456] |
| ECE (15 bins) | 0.0251 [0.0221, 0.0493] |

## Frozen thresholds on test

| Threshold | Precision | Recall | F1 | Expected cost |
| :--- | ---: | ---: | ---: | ---: |
| cost_optimal | 0.310 | 0.995 | 0.472 | 12.87 |
| max_precision_recall_0.80 | 0.472 | 0.845 | 0.605 | 37.13 |

## All learners (test point metrics)

| Model | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Calib |
| :--- | ---: | ---: | ---: | :--- |
| majority | 0.2654 | 0.5000 | 0.2654 | n/a |
| logistic | 0.6384 | 0.8421 | 0.6183 | isotonic |
| xgboost | 0.6455 | 0.8465 | 0.6372 | isotonic |
| lightgbm | 0.6378 | 0.8411 | 0.6189 | isotonic |

## Segments (test, tenure x MonthlyCharges tertiles from train)

| Segment | n | Churn rate | Mean risk | Flagged cost-opt | Flagged P@R>=.8 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| high__high | 209 | 0.144 | 0.156 | 193 | 60 |
| high__low | 135 | 0.015 | 0.026 | 35 | 3 |
| high__mid | 119 | 0.050 | 0.065 | 79 | 6 |
| low__high | 81 | 0.679 | 0.728 | 81 | 81 |
| low__low | 204 | 0.358 | 0.286 | 194 | 122 |
| low__mid | 195 | 0.528 | 0.540 | 195 | 173 |
| mid__high | 162 | 0.377 | 0.400 | 162 | 140 |
| mid__low | 141 | 0.050 | 0.088 | 103 | 19 |
| mid__mid | 163 | 0.227 | 0.210 | 159 | 66 |

## SHAP global importance (test, selected model)

| Feature | Mean |SHAP| |
| :--- | ---: |
| Contract | 0.7893 |
| tenure | 0.3242 |
| MonthlyCharges | 0.2327 |
| OnlineSecurity | 0.2305 |
| InternetService | 0.2176 |
| TechSupport | 0.1491 |
| PaymentMethod | 0.1287 |
| TotalCharges | 0.0918 |
| PaperlessBilling | 0.0815 |
| MultipleLines | 0.0815 |
