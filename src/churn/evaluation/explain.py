"""SHAP global feature importance for the selected model (explanatory only).

Explains the base estimator inside the fitted pipeline on the test
matrix; the calibration step is a monotone transform and does not change
importance ordering. One-hot columns are aggregated back to their parent
feature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from churn.data.telco import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churn.modeling.learners import FittedLearner

TOP_N = 20


def _parent_feature(col: str) -> str:
    if col in NUMERIC_FEATURES:
        return col
    head = col.split("_", 1)[0]
    if head in CATEGORICAL_FEATURES:
        return head
    return col


def shap_importance(
    learner: FittedLearner, X_test_raw: Any, *, top_n: int = TOP_N
) -> tuple[pd.DataFrame, str]:
    """Mean |SHAP| per original feature on test; returns (df, explainer_name)."""
    import shap  # noqa: PLC0415

    pipe = learner.pipeline
    features = pipe.named_steps["features"]
    model = pipe.named_steps["model"]
    X_test_t = features.transform(X_test_raw)
    cols = [str(c) for c in features.get_feature_names_out()]

    model_name = type(model).__name__
    if model_name == "XGBClassifier" or model_name == "LGBMClassifier":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        explainer_name = "TreeExplainer"
    else:
        # Linear model: exact linear attributions with test as background.
        background = X_test_t[:: max(1, len(X_test_t) // 100)]
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_test_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        explainer_name = "LinearExplainer"

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
    mean_abs = np.abs(arr).mean(axis=0)
    frame = pd.DataFrame(
        {"feature": [_parent_feature(c) for c in cols], "mean_abs_shap": mean_abs}
    )
    grouped = (
        frame.groupby("feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return grouped.head(top_n), explainer_name
