"""Render image.png from curated results/ artifacts only.

Panels: per-learner ROC-AUC/PR-AUC with bootstrap CIs, frozen-threshold
P/R/F1 for the selected model, the tenure x MonthlyCharges segment grid
(churn rate with mean predicted risk), and SHAP global importance.
Requires Matplotlib; run with
`uv run --with matplotlib python scripts/render_figure.py`.
Set CHURN_RESULTS_DIR to render from a different results directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("CHURN_RESULTS_DIR", ROOT / "results"))
OUT = ROOT / "image.png"

MODEL_COLORS = {
    "majority": "#9e9e9e",
    "logistic": "#1f77b4",
    "xgboost": "#2ca02c",
    "lightgbm": "#d62728",
}
ORDER = ["majority", "logistic", "xgboost", "lightgbm"]


def _ci(entry: dict) -> tuple[float, float, float]:
    return entry["mean"], entry["ci_low"], entry["ci_high"]


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    segments = pd.read_csv(RESULTS / "segments.csv")
    shap = pd.read_csv(RESULTS / "shap_importance.csv")
    sel = metrics["selected_model"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle(
        f"Telco churn benchmark ({metrics['protocol']}): "
        f"selected model {sel} (validation PR-AUC)",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.935,
        "Expected costs assume FN=780, FP=20 (declared assumptions); "
        "retention actions are recommendations, not measured savings.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    # Panel A: ROC-AUC / PR-AUC per learner with CIs.
    ax = axes[0, 0]
    names = [n for n in ORDER if n in metrics["models"]]
    x = np.arange(len(names))
    for j, key in enumerate(("roc_auc", "pr_auc")):
        off = -0.18 if key == "roc_auc" else 0.18
        means, los, his = [], [], []
        for n in names:
            m, lo, hi = _ci(metrics["models"][n]["test"]["ci"][key])
            means.append(m)
            los.append(m - lo)
            his.append(hi - m)
        ax.errorbar(
            x + off,
            means,
            yerr=[los, his],
            fmt="o",
            capsize=3,
            label=key.upper().replace("_", "-"),
            color=["#1f77b4", "#9467bd"][j],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Test discrimination (95% bootstrap CI)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: frozen thresholds on the selected model.
    ax = axes[0, 1]
    pt = metrics["models"][sel]["test"]["point"]
    groups = ["cost_optimal", "max_precision_recall_0.80"]
    labels = ["cost-optimal\n(FN 780 / FP 20)", "max precision\nrecall >= 0.80"]
    width = 0.26
    xpos = np.arange(len(groups))
    for j, (key, color) in enumerate(
        [("precision", "#1f77b4"), ("recall", "#2ca02c"), ("f1", "#d62728")]
    ):
        vals = [pt[f"{g}_{key}"] for g in groups]
        bars = ax.bar(
            xpos + (j - 1) * width, vals, width, label=key, color=color
        )
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}",
                ha="center",
                fontsize=7,
            )
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("score")
    ax.set_title(f"Frozen thresholds on test ({sel})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: segment grid, churn rate heatmap with mean risk annotated.
    ax = axes[1, 0]
    seg = segments.copy()
    seg[["tenure_band", "charge_band"]] = seg["segment"].str.split("__", expand=True)
    band_order = {"low": 0, "mid": 1, "high": 2}
    tenure_lv = sorted(seg["tenure_band"].unique(), key=band_order.get)
    charge_lv = sorted(seg["charge_band"].unique(), key=band_order.get)
    grid = np.full((len(tenure_lv), len(charge_lv)), np.nan)
    risk = np.full_like(grid, np.nan, dtype=float)
    counts = np.zeros_like(grid, dtype=int)
    for _, row in seg.iterrows():
        i = tenure_lv.index(row["tenure_band"])
        j = charge_lv.index(row["charge_band"])
        grid[i, j] = row["churn_rate"]
        risk[i, j] = row["mean_predicted_risk"]
        counts[i, j] = row["n"]
    vmax = max(0.5, float(np.nanmax(grid)))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0.0, vmax=vmax)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j, i,
                f"{grid[i, j]:.2f}\n({risk[i, j]:.2f})\nn={counts[i, j]}",
                ha="center", va="center", fontsize=7,
                color="white" if grid[i, j] > 0.6 * vmax else "black",
            )
    ax.set_xticks(range(len(charge_lv)))
    ax.set_xticklabels(charge_lv, fontsize=8)
    ax.set_xlabel("MonthlyCharges tertile (train cut points)")
    ax.set_yticks(range(len(tenure_lv)))
    ax.set_yticklabels([f"tenure {v}" for v in tenure_lv], fontsize=8)
    ax.set_ylabel("tenure tertile")
    ax.set_title("Test churn rate (mean predicted risk, n)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Panel D: SHAP top features.
    ax = axes[1, 1]
    top = shap.head(10).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"SHAP importance ({sel}, test)")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT, dpi=160)

    figure_data = {
        "source": "results/ (unedited kernel output)",
        "protocol": metrics["protocol"],
        "selected_model": sel,
        "discrimination": {
            n: {
                key: metrics["models"][n]["test"]["ci"][key]
                for key in ("roc_auc", "pr_auc")
            }
            for n in names
        },
        "frozen_thresholds_point": {
            g: {k: pt[f"{g}_{k}"] for k in ("precision", "recall", "f1", "expected_cost")}
            for g in groups
        },
        "segments": segments.to_dict(orient="records"),
        "shap_top10": shap.head(10).to_dict(orient="records"),
        "note": (
            "Expected costs assume FN=780, FP=20 (declared assumptions, "
            "not measured costs); retention actions are recommendations."
        ),
    }
    (Path(__file__).resolve().parent / "figure-data.json").write_text(
        json.dumps(figure_data, indent=2) + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
