
# Customer Segmentation and Retention - Project Summary

## 1. Objective
Build a full customer segmentation and churn prediction workflow from the Telco dataset, generate business-ready outputs, and improve model quality across key churn metrics.

## 2. Dataset Used
- File: WA_Fn-UseC_-Telco-Customer-Churn.csv
- Target: Churn (Yes/No mapped to 1/0)
- Rows used after cleaning: 7032

## 3. What Was Implemented
### Core End-to-End Pipeline
- Script created: customer_segmentation_retention.py
- Implemented complete flow:
  1. Data load and cleaning
  2. Feature encoding
  3. RFM scoring and segment labels
  4. K-Means clustering with elbow plot
  5. Churn model training and evaluation
  6. SHAP explainability
  7. Segment churn analysis
  8. Business recommendation export

### Visual and Report Outputs Generated
- elbow_curve.png
- cluster_scatter.png
- shap_summary.png
- churn_by_segment.png
- recommendation.txt

### Environment and Execution Setup
- Created local virtual environment: .venv
- Installed required ML libraries
- Resolved macOS XGBoost dependency by installing libomp

### Colab/Notebook Support
- Created runbook notebook: telco_churn_runbook.ipynb
- Notebook automates package checks, pipeline run, result display, and ensemble search run

### Additional Model Improvement Work
Created dedicated optimization scripts and retained the final one:
- model_search_ensemble.py (active)

## 4. Baseline Model Results (Initial Full Pipeline)
- ROC-AUC: 0.8421
- F1 (churn class): 0.6185
- Accuracy: 0.7300
- Precision (churn class): 0.5000
- Recall (churn class): 0.8200

## 5. Newest Ensemble Results (Latest)
Model candidate:
- name: blend2:xgb_balance:rf:0.50

Scores:
- roc_auc: 0.8411601637927019
- threshold: 0.49499999999999994
- precision: 0.5296167247386759
- recall: 0.8128342245989305
- f1: 0.6413502109704642
- accuracy: 0.7583511016346838
- score: 5.332506797669769
- all_non_auc_better: false
- all_better: false

## 6. Baseline vs Newest Ensemble (Delta)
- ROC-AUC: 0.8421 -> 0.8411601637927019 (delta: -0.0009398362)
- F1: 0.6185 -> 0.6413502110 (delta: +0.0228502110)
- Accuracy: 0.7300 -> 0.7583511016 (delta: +0.0283511016)
- Precision: 0.5000 -> 0.5296167247 (delta: +0.0296167247)
- Recall: 0.8200 -> 0.8128342246 (delta: -0.0071657754)

Interpretation:
- Strong improvements in F1, accuracy, and precision.
- Recall is slightly lower than baseline.
- ROC-AUC is lower by about 0.00094, which is a very small drop.
- This is why all_non_auc_better is false (recall did not exceed baseline) and all_better is false (ROC-AUC and recall did not both exceed baseline).

## 7. What Was Done to Improve Scores
- Expanded model search space for XGBoost (depth, estimators, learning rate, regularization, sampling, class weight).
- Added probability-threshold sweeps to optimize classification trade-offs.
- Tested multiple model families: XGBoost, Logistic Regression, Random Forest, Extra Trees, CatBoost.
- Built weighted blend/ensemble search to improve practical churn metrics.
- Added feature interactions and cluster/RFM context in search variants.

## 8. Recommended Next Tuning Direction
To make all_better true, focus on recovering small ROC-AUC and recall gaps while keeping F1/accuracy gains:
- Add constrained threshold optimization with hard minimum recall >= 0.82.
- Blend with an AUC-strong XGBoost probability stream at small weight.
- Run larger AUC-first search with early stopping and then constrained thresholding.

## 9. Main Files to Use
- Primary pipeline: customer_segmentation_retention.py
- Notebook runner: telco_churn_runbook.ipynb
- Ensemble search: model_search_ensemble.py
- Summary document: PROJECT_SUMMARY.md

## 10. Four-Lever Implementation Update (Latest Run)
Implemented in model_search_ensemble.py exactly in this order:
- Lever 1: Feature engineering (10 engineered features)
- Lever 2: Probability calibration (CalibratedClassifierCV, isotonic, cv=3)
- Lever 3: Cost-based threshold optimization (FN cost=200, FP cost=20)
- Lever 4: Optuna Bayesian tuning (80 trials)

Optuna outcome:
- Best CV AUC: 0.8510142768064656

Feature-engineering verification (top-20 importance check):
- Engineered features in top 20:
  - mtm_high_charge
  - early_mtm
  - charges_per_tenure
  - fiber_no_security
  - digital_risk
  - log_total
  - log_monthly

Holdout results from latest script run:
- Calibrated model at threshold 0.50:
  - roc_auc: 0.8421631611370235
  - precision: 0.6366459627329193
  - recall: 0.5481283422459893
  - f1: 0.5890804597701149
  - accuracy: 0.7967306325515281
- Recall-constrained threshold (minimum recall 0.80):
  - threshold: 0.28
  - roc_auc: 0.8421631611370235
  - precision: 0.5067340067340067
  - recall: 0.8048128342245989
  - f1: 0.621900826446281
  - accuracy: 0.7398720682302772
- Cost-optimized threshold (cost_fn=200, cost_fp=20):
  - threshold: 0.11
  - roc_auc: 0.8421631611370235
  - precision: 0.3899233296823658
  - recall: 0.9518716577540107
  - f1: 0.5532245532245532
  - accuracy: 0.5913290689410092
  - business_cost: 14740.0

LightGBM comparison (same calibration and threshold workflow):
- Recall-constrained ROC-AUC:
  - calibrated_xgb: 0.8421631611370235
  - calibrated_lgbm: 0.8194850676343758
- Final selected model remains calibrated_xgb.

Interpretation:
- The recall-constrained threshold remains the best campaign operating point in this run.
- Cost optimization still pushes threshold low and should not be deployed alone.
- New engineered features (fiber_no_security and early_mtm) were validated by appearing in the top-20 XGBoost importance list.
