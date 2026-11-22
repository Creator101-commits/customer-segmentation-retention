"""Test metrics with bootstrap confidence intervals.

All metrics are computed once on the frozen test partition; 2,000
bootstrap resamples (seed 0) give percentile CIs. Thresholds are frozen
on validation before this module ever sees test labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from churn.modeling.thresholds import (
    COST_FN_DEFAULT,
    COST_FP_DEFAULT,
    _metrics_at,
    expected_cost,
)

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
ECE_BINS = 15


def expected_calibration_error(
    y: np.ndarray, probs: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """ECE over equal-width probability bins on [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (probs >= lo) & (probs <= hi) if hi == 1.0 else (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(probs[mask].mean())
        ece += (n / len(y)) * abs(acc - conf)
    return float(ece)


def point_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    thresholds: dict[str, float] | None = None,
    cost_fn: float = COST_FN_DEFAULT,
    cost_fp: float = COST_FP_DEFAULT,
) -> dict[str, float]:
    """Scalar test metrics; optionally P/R/F1/cost at named thresholds."""
    out: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "log_loss": float(log_loss(y, probs)),
        "brier": float(brier_score_loss(y, probs)),
        "ece": expected_calibration_error(y, probs),
        "n": float(len(y)),
        "positive_rate": float(y.mean()),
    }
    for name, thr in (thresholds or {}).items():
        m = _metrics_at(y, probs, thr)
        out[f"{name}_precision"] = m["precision"]
        out[f"{name}_recall"] = m["recall"]
        out[f"{name}_f1"] = m["f1"]
        out[f"{name}_expected_cost"] = float(
            expected_cost(y, probs, thr, cost_fn, cost_fp)
        )
        out[f"{name}_threshold"] = float(thr)
    return out


def _bootstrap_stats(
    y: np.ndarray, probs: np.ndarray, idx: np.ndarray, thresholds: dict[str, float]
) -> dict[str, float]:
    yb = y[idx]
    pb = probs[idx]
    out = point_metrics(yb, pb, thresholds=thresholds)
    # PR-AUC/ROC-AUC undefined on single-class resamples; signal via nan.
    return out


def bootstrap_cis(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    thresholds: dict[str, float] | None = None,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Percentile 95% CIs for every point metric via bootstrap resampling."""
    thresholds = thresholds or {}
    point = point_metrics(y, probs, thresholds=thresholds)
    rng = np.random.default_rng(seed)
    n = len(y)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats = _bootstrap_stats(y, probs, idx, thresholds)
        for k, v in stats.items():
            samples[k].append(v)
    cis: dict[str, dict[str, float]] = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        cis[k] = {
            "mean": float(point[k]),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }
    return cis


def classification_report_at(
    y: np.ndarray, probs: np.ndarray, thr: float
) -> dict[str, float]:
    """Precision/recall/F1 at a threshold via sklearn's summary."""
    pred = (probs >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}
