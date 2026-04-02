# One-Page Results Summary

## 1) Model Card
Selected production operating point:
- Model: Calibrated XGBoost (isotonic calibration, cv=3)
- Threshold: 0.28 (recall-constrained)
- ROC-AUC: 0.8422
- Precision: 0.5067
- Recall: 0.8048
- F1: 0.6219
- Accuracy: 0.7399

What these mean in plain English:
- ROC-AUC 0.8422: the model ranks true churners above non-churners well, but still has room to improve.
- Precision 0.5067: about 51 out of 100 flagged customers are real churners.
- Recall 0.8048: the campaign catches about 80 out of 100 actual churners.
- F1 0.6219: this is the balanced quality score between precision and recall at the selected threshold.
- Threshold 0.28: lower than 0.50 by design to protect churn capture for retention campaigns.

## 2) Segment Table

| Segment | Customer Count | Churn Rate |
|---|---:|---:|
| Champion | 266 | 58.65% |
| Potential Loyalist | 2,750 | 34.44% |
| At Risk | 2,817 | 25.59% |
| Lost | 1,199 | 3.75% |

Notes:
- Segment counts and rates are computed from the cleaned 7,032-row dataset.
- Campaign scoring threshold remains 0.28 across segments.

## 3) Top 5 Churn Drivers (SHAP-Aligned, Plain Language)
1. Customers in month-to-month contracts with tenure <= 12 months churn at 51.35%, which is about 1.93x the base churn rate.
2. Fiber-internet customers without online security churn at 49.36%, about 1.86x the base rate.
3. Customers with electronic-check payment plus paperless billing churn at 49.77%, about 1.87x the base rate.
4. Customers with high monthly charges and short tenure churn at 68.96%, about 2.59x the base rate.
5. Month-to-month customers overall churn at 42.71%, about 1.61x the base rate.

Base churn rate reference: 26.58%.

## 4) Budget Recommendation
Assumption for campaign sizing (requested planning scenario):
- Predicted churners to target: 1,406
- Expected true churners among flagged customers (using ~50.7% precision): ~703
- Offer cost per targeted customer: $20
- Estimated retention spend: 703 x $20 = $14,060
- Estimated annual revenue at risk per churner: $780
- Estimated recoverable annual revenue: 703 x $780 = $548,340
- Implied ROI multiple: $548,340 / $14,060 = 39.0x

Recommendation:
- Use threshold 0.28 as the primary campaign threshold.
- Budget for at least $14,060 for this campaign size.
- Keep the 0.11 cost-only threshold out of production because precision is too low for efficient spend.
