"""Frozen-plan churn benchmark.

Protocol: all preprocessing fits on train; Optuna tunes on train CV;
calibration and model selection use validation only; thresholds freeze
on validation; the analysis plan (with card_write_epoch) is persisted
before the first test read; test metrics, bootstrap CIs, segmentation,
and SHAP run once afterward. manifest.json records both epochs and
ordering_ok = card_write_epoch < first_test_read_epoch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from churn.data.telco import FEATURES, data_manifest, load_telco
from churn.evaluation.explain import shap_importance
from churn.evaluation.metrics import bootstrap_cis, point_metrics
from churn.evaluation.segmentation import segment_table, tertile_cuts
from churn.modeling.learners import (
    LEARNER_NAMES,
    N_TRIALS,
    SEED,
    FittedLearner,
    fit_baseline,
    fit_learner,
)
from churn.modeling.thresholds import (
    COST_FN_DEFAULT,
    COST_FP_DEFAULT,
    MIN_RECALL,
    frozen_thresholds,
)

PROTOCOL = "churn-benchmark-v1"


def _epoch() -> float:
    return time.time()


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def run_benchmark(
    data_path: Path,
    out_dir: Path,
    *,
    n_trials: int = N_TRIALS,
    n_boot: int = 2000,
    seed: int = SEED,
    git_sha: str | None = None,
    smoke: bool = False,
    verify: bool = True,
) -> dict[str, Any]:
    """Run the full frozen protocol and write benchmark artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_telco(data_path, verify=verify)
    df = data.df
    y = data.y
    X = df[FEATURES]
    X_train = X.iloc[data.train_idx]
    X_val = X.iloc[data.val_idx]
    y_train = y[data.train_idx]
    y_val = y[data.val_idx]

    # Fit every learner on train, calibrate and score on validation.
    learners: dict[str, FittedLearner] = {
        "majority": fit_baseline(X_train, y_train)
    }
    for name in LEARNER_NAMES:
        learners[name] = fit_learner(
            name, X_train, y_train, X_val, y_val, n_trials=n_trials
        )

    probs_val = {
        name: learner.predict_proba(X_val) for name, learner in learners.items()
    }

    # Validation PR-AUC selects the model; thresholds freeze per learner.
    val_pr_auc = {
        name: float(average_precision_score(y_val, p))
        for name, p in probs_val.items()
    }
    val_roc_auc = {
        name: float(roc_auc_score(y_val, p))
        for name, p in probs_val.items()
    }
    selected = max(
        (n for n in learners if n != "majority"),
        key=lambda n: val_pr_auc[n],
    )
    thresholds = {
        name: frozen_thresholds(
            y_val,
            probs_val[name],
            cost_fn=COST_FN_DEFAULT,
            cost_fp=COST_FP_DEFAULT,
            min_recall=MIN_RECALL,
        )
        for name in learners
    }
    frozen_test_thresholds = {
        name: {
            "cost_optimal": t["cost_optimal"]["threshold"],
            "max_precision_recall_0.80": t["max_precision_recall_0.80"][
                "threshold"
            ],
        }
        for name, t in thresholds.items()
    }

    # Freeze the plan before the first test read.
    card_write_epoch = _epoch()
    plan = {
        "protocol": PROTOCOL,
        "seed": seed,
        "n_trials": n_trials,
        "n_bootstrap": n_boot,
        "costs": {
            "cost_fn": COST_FN_DEFAULT,
            "cost_fp": COST_FP_DEFAULT,
            "min_recall": MIN_RECALL,
            "note": "declared assumptions, not measured costs",
        },
        "split": "stratified 60/20/20",
        "learners": sorted(learners),
        "selected_model": selected,
        "selection_rule": "validation PR-AUC",
        "val_pr_auc": val_pr_auc,
        "frozen_test_thresholds": frozen_test_thresholds,
        "calibration": {
            name: learners[name].calibration_method for name in learners
        },
        "val_logloss_by_method": {
            name: learners[name].val_logloss_by_method for name in learners
        },
        "tuned_params": {
            name: learners[name].tuned_params for name in learners
        },
        "smoke": smoke,
    }
    _write_json(out_dir / "analysis_plan.json", plan)

    # First and only test read.
    first_test_read_epoch = _epoch()
    X_test = X.iloc[data.test_idx]
    y_test = y[data.test_idx]

    test_results: dict[str, Any] = {}
    for name, learner in learners.items():
        probs_test = learner.predict_proba(X_test)
        thr_map = {
            "cost_optimal": frozen_test_thresholds[name]["cost_optimal"],
            "max_precision_recall_0.80": frozen_test_thresholds[name][
                "max_precision_recall_0.80"
            ],
        }
        point = point_metrics(y_test, probs_test, thresholds=thr_map)
        cis = bootstrap_cis(
            y_test, probs_test, thresholds=thr_map, n_boot=n_boot, seed=0
        )
        test_results[name] = {
            "point": point,
            "ci": cis,
            "probs": probs_test,
        }

    sel = test_results[selected]
    sel_thresholds = {
        "cost_optimal": frozen_test_thresholds[selected]["cost_optimal"],
        "max_precision_recall_0.80": frozen_test_thresholds[selected][
            "max_precision_recall_0.80"
        ],
    }

    # Segmentation on test for the selected model.
    cuts = tertile_cuts(X_train.reset_index(drop=True))
    test_df = df.iloc[data.test_idx].reset_index(drop=True)
    segments = segment_table(
        test_df,
        y_test,
        sel["probs"],
        cuts,
        sel_thresholds,
    )
    seg_frame = pd.DataFrame(segments)
    seg_frame.to_csv(out_dir / "segments.csv", index=False)

    # SHAP global importance on test for the selected model.
    shap_df, explainer_name = shap_importance(learners[selected], X_test)
    shap_df.to_csv(out_dir / "shap_importance.csv", index=False)

    metrics = {
        "protocol": PROTOCOL,
        "selected_model": selected,
        "selection_rule": "validation PR-AUC",
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "models": {
            name: {
                "calibration_method": learners[name].calibration_method,
                "is_baseline": learners[name].is_baseline,
                "tuned_params": learners[name].tuned_params,
                "val_logloss_by_method": learners[name].val_logloss_by_method,
                "thresholds_validation": thresholds[name],
                "test": {
                    "point": test_results[name]["point"],
                    "ci": test_results[name]["ci"],
                },
            }
            for name in learners
        },
        "segments": segments,
        "shap": {
            "explainer": explainer_name,
            "top_features": shap_df.to_dict(orient="records"),
        },
        "smoke": smoke,
    }
    _write_json(out_dir / "metrics.json", metrics)

    manifest = {
        "git_sha": git_sha or _git_sha(),
        "protocol": PROTOCOL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "card_write_epoch": card_write_epoch,
        "first_test_read_epoch": first_test_read_epoch,
        "ordering_ok": bool(card_write_epoch < first_test_read_epoch),
        "data": data_manifest(data, data_path),
        "config": {
            "n_trials": n_trials,
            "n_bootstrap": n_boot,
            "bootstrap_seed": 0,
            "seed": seed,
            "cost_fn": COST_FN_DEFAULT,
            "cost_fp": COST_FP_DEFAULT,
            "min_recall": MIN_RECALL,
            "ece_bins": 15,
            "smoke": smoke,
        },
        "artifacts": [
            "analysis_plan.json",
            "metrics.json",
            "manifest.json",
            "report.md",
            "segments.csv",
            "shap_importance.csv",
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    (out_dir / "report.md").write_text(
        render_report(metrics, manifest), encoding="utf-8"
    )
    return {"metrics": metrics, "manifest": manifest}


def _git_sha() -> str:
    import subprocess  # noqa: PLC0415

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return "unknown"


def render_report(metrics: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Markdown report from metrics + manifest only."""
    sel = metrics["selected_model"]
    m = metrics["models"][sel]
    pt = m["test"]["point"]
    ci = m["test"]["ci"]

    def _f(key: str) -> str:
        c = ci.get(key, {})
        return f"{c.get('mean', 0.0):.4f} [{c.get('ci_low', 0.0):.4f}, {c.get('ci_high', 0.0):.4f}]"

    lines = [
        "# Churn Benchmark Report",
        "",
        f"- git SHA: `{manifest['git_sha']}`",
        f"- Protocol: {metrics['protocol']}",
        f"- Selected model (validation PR-AUC): **{sel}** "
        f"(PR-AUC {metrics['val_pr_auc'][sel]:.4f})",
        f"- ordering_ok: {manifest['ordering_ok']}",
        "",
        "## Test metrics (frozen, 95% bootstrap CI, 2000 resamples)",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| ROC-AUC | {_f('roc_auc')} |",
        f"| PR-AUC | {_f('pr_auc')} |",
        f"| Log loss | {_f('log_loss')} |",
        f"| Brier | {_f('brier')} |",
        f"| ECE (15 bins) | {_f('ece')} |",
        "",
        "## Frozen thresholds on test",
        "",
        "| Threshold | Precision | Recall | F1 | Expected cost |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for tname in ("cost_optimal", "max_precision_recall_0.80"):
        lines.append(
            f"| {tname} | {pt[f'{tname}_precision']:.3f} | "
            f"{pt[f'{tname}_recall']:.3f} | {pt[f'{tname}_f1']:.3f} | "
            f"{pt[f'{tname}_expected_cost']:.2f} |"
        )
    lines += [
        "",
        "## All learners (test point metrics)",
        "",
        "| Model | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Calib |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    for name, mm in metrics["models"].items():
        lines.append(
            f"| {name} | {metrics['val_pr_auc'][name]:.4f} | "
            f"{mm['test']['point']['roc_auc']:.4f} | "
            f"{mm['test']['point']['pr_auc']:.4f} | "
            f"{mm['calibration_method'] or 'n/a'} |"
        )
    lines += [
        "",
        "## Segments (test, tenure x MonthlyCharges tertiles from train)",
        "",
        "| Segment | n | Churn rate | Mean risk | Flagged cost-opt | Flagged P@R>=.8 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["segments"]:
        lines.append(
            f"| {row['segment']} | {row['n']} | {row['churn_rate']:.3f} | "
            f"{row['mean_predicted_risk']:.3f} | "
            f"{row.get('flagged_cost_optimal', 0)} | "
            f"{row.get('flagged_max_precision_recall_0.80', 0)} |"
        )
    lines += [
        "",
        "## SHAP global importance (test, selected model)",
        "",
        "| Feature | Mean |SHAP| |",
        "| :--- | ---: |",
    ]
    for row in metrics["shap"]["top_features"][:10]:
        lines.append(f"| {row['feature']} | {row['mean_abs_shap']:.4f} |")
    lines.append("")
    return "\n".join(lines)
