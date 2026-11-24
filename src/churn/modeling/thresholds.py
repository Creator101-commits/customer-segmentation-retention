"""Decision thresholds on validation predictions only.

Two frozen thresholds per learner: the cost-optimal threshold under
declared FN/FP costs, and the max-precision threshold subject to
recall >= 0.80. A cost-ratio sensitivity table varies the assumed
FN/FP ratio on validation. Nothing here touches test data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

COST_FN_DEFAULT = 780.0
COST_FP_DEFAULT = 20.0
MIN_RECALL = 0.80
SENSITIVITY_RATIOS = (5.0, 10.0, 20.0, 39.0, 80.0)
N_THRESHOLD_GRID = 1000


def _threshold_grid(probs: np.ndarray) -> np.ndarray:
    """Candidate cutoffs: quantiles of predicted probabilities plus 0/1."""
    qs = np.linspace(0.0, 1.0, N_THRESHOLD_GRID)
    grid = np.unique(np.quantile(probs, qs))
    return np.concatenate([[0.0], grid, [1.0]])


def _metrics_at(y: np.ndarray, probs: np.ndarray, thr: float) -> dict[str, float]:
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def expected_cost(
    y: np.ndarray, probs: np.ndarray, thr: float, cost_fn: float, cost_fp: float
) -> float:
    """Mean expected cost per customer under declared FN/FP costs."""
    pred = (probs >= thr).astype(int)
    fn = float(((pred == 0) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    return (fn * cost_fn + fp * cost_fp) / len(y)


def cost_optimal_threshold(
    y: np.ndarray,
    probs: np.ndarray,
    cost_fn: float = COST_FN_DEFAULT,
    cost_fp: float = COST_FP_DEFAULT,
) -> dict[str, Any]:
    "" "Threshold minimizing total expected cost on the given labels."""
    grid = _threshold_grid(probs)
    costs = np.array(
        [
            expected_cost(y, probs, t, cost_fn, cost_fp) * len(y)
            for t in grid
        ]
    )
    best_idx = int(np.argmin(costs))
    thr = float(grid[best_idx])
    out = _metrics_at(y, probs, thr)
    out["expected_cost_per_customer"] = float(costs[best_idx] / len(y))
    out["cost_fn"] = float(cost_fn)
    out["cost_fp"] = float(cost_fp)
    return out


def max_precision_threshold(
    y: np.ndarray,
    probs: np.ndarray,
    min_recall: float = MIN_RECALL,
) -> dict[str, Any]:
    """Highest-precision threshold whose recall meets min_recall.

    Ties on precision break toward higher recall, then the lower
    threshold (more coverage).
    """
    grid = _threshold_grid(probs)
    candidates: list[tuple[float, float, float, float]] = []
    for thr in grid:
        m = _metrics_at(y, probs, float(thr))
        if m["recall"] >= min_recall and (m["tp"] + m["fp"]) > 0:
            candidates.append(
                (m["precision"], m["recall"], -float(thr), float(thr))
            )
    if not candidates:
        # No threshold reaches the floor; keep the highest-recall one.
        recalls = [
            (_metrics_at(y, probs, float(t))["recall"], float(t)) for t in grid
        ]
        best_recall, thr = max(recalls, key=lambda rt: (rt[0], -rt[1]))
        out = _metrics_at(y, probs, thr)
        out["recall_floor_met"] = False
        out["best_available_recall"] = float(best_recall)
        return out
    candidates.sort(reverse=True)
    _, _, _, thr = candidates[0]
    out = _metrics_at(y, probs, thr)
    out["recall_floor_met"] = True
    out["min_recall"] = float(min_recall)
    return out


def cost_sensitivity_table(
    y: np.ndarray,
    probs: np.ndarray,
    cost_fp: float = COST_FP_DEFAULT,
    ratios: tuple[float, ...] = SENSITIVITY_RATIOS,
) -> list[dict[str, Any]]:
    """Cost-optimal threshold per FN/FP ratio (cost_fn = ratio * cost_fp)."""
    rows = []
    for ratio in ratios:
        cost_fn = ratio * cost_fp
        row = cost_optimal_threshold(y, probs, cost_fn=cost_fn, cost_fp=cost_fp)
        row["fn_fp_ratio"] = float(ratio)
        rows.append(row)
    return rows


def frozen_thresholds(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    *,
    cost_fn: float = COST_FN_DEFAULT,
    cost_fp: float = COST_FP_DEFAULT,
    min_recall: float = MIN_RECALL,
) -> dict[str, Any]:
    """Both frozen thresholds plus the sensitivity table, all on validation."""
    return {
        "cost_optimal": cost_optimal_threshold(
            y_val, probs_val, cost_fn=cost_fn, cost_fp=cost_fp
        ),
        "max_precision_recall_0.80": max_precision_threshold(
            y_val, probs_val, min_recall=min_recall
        ),
        "cost_sensitivity": cost_sensitivity_table(
            y_val, probs_val, cost_fp=cost_fp
        ),
        "declared_costs": {
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "note": (
                "assumption: FN = lost customer value proxy, "
                "FP = retention-offer cost; not measured"
            ),
        },
    }
