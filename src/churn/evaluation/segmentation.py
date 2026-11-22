"""Value segmentation: tenure x MonthlyCharges tertiles.

Tertile cut points are computed on the TRAIN split only and applied
unchanged to validation and test. Per segment on test: n, churn rate,
mean predicted risk, and counts flagged at each frozen threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SEGMENT_FEATURES = ("tenure", "MonthlyCharges")
N_BUCKETS = 3
BUCKET_LABELS = ("low", "mid", "high")


def tertile_cuts(train_df: pd.DataFrame) -> dict[str, list[float]]:
    """Tertile cut points per segment feature, from train only."""
    cuts: dict[str, list[float]] = {}
    for col in SEGMENT_FEATURES:
        values = train_df[col].to_numpy(dtype=float)
        q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
        cuts[col] = [float(q1), float(q2)]
    return cuts


def assign_segments(df: pd.DataFrame, cuts: dict[str, list[float]]) -> pd.Series:
    """Segment label like ``tenure_low__charges_high`` for every row."""
    tenure_bucket = pd.cut(
        df["tenure"],
        bins=[-np.inf, *cuts["tenure"], np.inf],
        labels=BUCKET_LABELS,
    )
    charge_bucket = pd.cut(
        df["MonthlyCharges"],
        bins=[-np.inf, *cuts["MonthlyCharges"], np.inf],
        labels=BUCKET_LABELS,
    )
    return tenure_bucket.astype(str) + "__" + charge_bucket.astype(str)


def segment_table(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    probs_test: np.ndarray,
    cuts: dict[str, list[float]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Per-segment test stats for the selected model."""
    segments = assign_segments(test_df, cuts)
    rows: list[dict[str, Any]] = []
    for seg in sorted(segments.unique()):
        mask = (segments == seg).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        row: dict[str, Any] = {
            "segment": str(seg),
            "n": n,
            "churn_rate": float(y_test[mask].mean()),
            "mean_predicted_risk": float(probs_test[mask].mean()),
        }
        for name, thr in thresholds.items():
            row[f"flagged_{name}"] = int((probs_test[mask] >= thr).sum())
        rows.append(row)
    return rows
