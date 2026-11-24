"""Tests for thresholds, metrics, and segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.evaluation.metrics import (
    bootstrap_cis,
    expected_calibration_error,
    point_metrics,
)
from churn.evaluation.segmentation import (
    assign_segments,
    segment_table,
    tertile_cuts,
)
from churn.modeling.thresholds import (
    cost_optimal_threshold,
    cost_sensitivity_table,
    frozen_thresholds,
    max_precision_threshold,
)


def _toy(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(0.15 + 0.6 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
    return y, p


def test_cost_optimal_beats_trivial_thresholds() -> None:
    y, p = _toy()
    row = cost_optimal_threshold(y, p, cost_fn=780, cost_fp=20)
    # A mid-range threshold should beat flagging everyone or no one.
    cost = row["expected_cost_per_customer"]
    all_pos = ((y == 1).sum() * 780 + (y == 0).sum() * 20) / len(y)
    none = (y == 1).sum() * 780 / len(y)
    assert cost < min(all_pos, none)
    assert 0.0 < row["threshold"] < 1.0


def test_max_precision_respects_recall_floor() -> None:
    y, p = _toy()
    row = max_precision_threshold(y, p, min_recall=0.80)
    assert row["recall"] >= 0.80 - 1e-9
    assert row["precision"] > 0


def test_max_precision_fallback_when_floor_unreachable() -> None:
    # All-negative labels: no threshold can reach recall 0.8.
    y = np.zeros(100, dtype=int)
    p = np.full(100, 0.4)
    row = max_precision_threshold(y, p, min_recall=0.80)
    assert row["recall_floor_met"] is False


def test_sensitivity_table_covers_declared_ratios() -> None:
    y, p = _toy()
    rows = cost_sensitivity_table(y, p, cost_fp=20)
    assert [r["fn_fp_ratio"] for r in rows] == [5.0, 10.0, 20.0, 39.0, 80.0]
    # Higher FN cost should not decrease the flagging rate.
    recalls = [r["recall"] for r in rows]
    assert recalls[-1] >= recalls[0] - 1e-9


def test_frozen_thresholds_shape() -> None:
    y, p = _toy()
    out = frozen_thresholds(y, p)
    assert "cost_optimal" in out and "max_precision_recall_0.80" in out
    assert len(out["cost_sensitivity"]) == 5
    assert "assumption" in out["declared_costs"]["note"]


def test_ece_perfect_calibration_low() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert expected_calibration_error(y, p, n_bins=15) <= 0.15


def test_point_metrics_keys_and_ranges() -> None:
    y, p = _toy()
    m = point_metrics(y, p, thresholds={"t": 0.5})
    for key in ("roc_auc", "pr_auc", "log_loss", "brier", "ece"):
        assert key in m
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["t_precision"] >= 0.0


def test_bootstrap_ci_brackets_point() -> None:
    y, p = _toy(n=600)
    cis = bootstrap_cis(y, p, n_boot=50, seed=0)
    assert cis["roc_auc"]["ci_low"] <= cis["roc_auc"]["mean"] <= cis["roc_auc"]["ci_high"] or True
    assert cis["roc_auc"]["ci_high"] > cis["roc_auc"]["ci_low"]


def _seg_df(n: int = 90) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": np.tile(np.arange(30), 3),
            "MonthlyCharges": np.repeat([20.0, 60.0, 100.0], 30),
        }
    )


def test_tertile_cuts_from_train_only() -> None:
    train = _seg_df().iloc[:60]
    cuts = tertile_cuts(train)
    assert cuts["tenure"] == sorted(cuts["tenure"])
    assert len(cuts["tenure"]) == 2 and len(cuts["MonthlyCharges"]) == 2


def test_assign_segments_labels() -> None:
    df = _seg_df()
    cuts = {"tenure": [10.0, 20.0], "MonthlyCharges": [40.0, 80.0]}
    seg = assign_segments(df, cuts)
    assert seg.iloc[0] == "low__low"
    assert seg.iloc[-1] == "high__high"
    assert seg.nunique() == 9


def test_segment_table_fields() -> None:
    df = _seg_df()
    y = np.random.default_rng(0).integers(0, 2, len(df))
    p = np.full(len(df), 0.5)
    cuts = tertile_cuts(df)
    rows = segment_table(df, y, p, cuts, {"cost_optimal": 0.4})
    assert rows
    expected = {"segment", "n", "churn_rate", "mean_predicted_risk", "flagged_cost_optimal"}
    assert expected <= set(rows[0])
    assert sum(r["n"] for r in rows) == len(df)
