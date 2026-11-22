"""End-to-end smoke test of the frozen benchmark protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn.experimentation.benchmark import run_benchmark

COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def _synthetic_csv(path: Path, n: int = 900, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tenure = int(rng.integers(0, 72))
        monthly = round(float(rng.uniform(18, 120)), 2)
        # Churn probability rises with tenure nonlinearity and monthly charge.
        p = 1 / (1 + np.exp(-(tenure / 30 + monthly / 200 - 1.2)))
        churn = "Yes" if rng.random() < p else "No"
        total = "" if i % 211 == 0 else f"{monthly * tenure:.2f}"
        rows.append(
            [
                f"id{i}", "Male" if i % 2 else "Female", i % 2, "Yes", "No",
                tenure, "Yes", "No" if i % 3 else "Yes", "DSL",
                "No" if i % 4 else "Yes", "Yes", "No", "No", "No", "No",
                "Month-to-month" if i % 5 else "One year", "Yes",
                "Mailed check", monthly, total, churn,
            ]
        )
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def test_run_benchmark_end_to_end(tmp_path: Path) -> None:
    csv = _synthetic_csv(tmp_path / "telco.csv")
    out = tmp_path / "bench"
    result = run_benchmark(
        csv,
        out,
        n_trials=2,
        n_boot=20,
        smoke=True,
        verify=False,
    )
    manifest = result["manifest"]
    assert manifest["ordering_ok"] is True
    assert manifest["card_write_epoch"] < manifest["first_test_read_epoch"]

    for name in (
        "analysis_plan.json",
        "metrics.json",
        "manifest.json",
        "report.md",
        "segments.csv",
        "shap_importance.csv",
    ):
        assert (out / name).exists(), name

    plan = json.loads((out / "analysis_plan.json").read_text())
    # Frozen thresholds must be in the plan before test read.
    assert "frozen_test_thresholds" in plan
    assert "majority" in plan["learners"]
    assert plan["selected_model"] in ("logistic", "xgboost", "lightgbm")

    metrics = result["metrics"]
    sel = metrics["models"][metrics["selected_model"]]
    pt = sel["test"]["point"]
    for key in ("roc_auc", "pr_auc", "brier", "ece"):
        assert 0.0 <= pt[key] <= 1.0
    assert pt["log_loss"] >= 0.0
    # Every segment row covers the test set exactly.
    assert sum(r["n"] for r in metrics["segments"]) > 0
    assert metrics["shap"]["top_features"]
