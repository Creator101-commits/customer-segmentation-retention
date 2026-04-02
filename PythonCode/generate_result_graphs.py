import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_report(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_metric_comparison(report, out_dir: Path):
    baseline = report["baseline_reference"]
    results = report["results"]

    metric_keys = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    metric_labels = ["ROC-AUC", "F1", "Precision", "Recall", "Accuracy"]

    names = ["Baseline"] + [r["name"] for r in results]
    values_by_model = []
    values_by_model.append([baseline[k] for k in metric_keys])
    for result in results:
        values_by_model.append([result[k] for k in metric_keys])

    x = np.arange(len(metric_keys))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for idx, values in enumerate(values_by_model):
        offset = (idx - (len(values_by_model) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=names[idx])

    plt.xticks(x, metric_labels)
    plt.ylim(0.35, 1.0)
    plt.ylabel("Score")
    plt.title("Churn Model Metrics: Baseline vs New Results")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "results_metrics_comparison.png", dpi=160)
    plt.close()


def plot_model_summary(results, out_dir: Path):
    names = [r["name"] for r in results]
    f1 = [r["f1"] for r in results]
    auc = [r["roc_auc"] for r in results]

    x = np.arange(len(names))

    fig, ax1 = plt.subplots(figsize=(11, 5))
    bars = ax1.bar(x, f1, color="#1f77b4", alpha=0.8, label="F1")
    ax1.set_ylabel("F1 Score", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0.45, 0.70)

    ax2 = ax1.twinx()
    ax2.plot(x, auc, color="#d62728", marker="o", linewidth=2.0, label="ROC-AUC")
    ax2.set_ylabel("ROC-AUC", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0.78, 0.86)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_title("Model-Level Performance (F1 and ROC-AUC)")

    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    plt.savefig(out_dir / "results_model_summary.png", dpi=160)
    plt.close()


def plot_cost_breakdown(results, out_dir: Path):
    cost_row = None
    for result in results:
        if "business_cost" in result:
            cost_row = result
            break

    if cost_row is None:
        return

    fp = int(cost_row.get("fp", 0))
    fn = int(cost_row.get("fn", 0))
    cost_fp = float(cost_row.get("cost_fp", 0.0))
    cost_fn = float(cost_row.get("cost_fn", 0.0))

    fp_cost_total = fp * cost_fp
    fn_cost_total = fn * cost_fn
    total = float(cost_row.get("business_cost", fp_cost_total + fn_cost_total))

    labels = ["FP Cost", "FN Cost", "Total Cost"]
    values = [fp_cost_total, fn_cost_total, total]
    colors = ["#ff7f0e", "#9467bd", "#2ca02c"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("USD")
    plt.title("Cost-Optimized Threshold: Business Cost Breakdown")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"${value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_dir / "results_business_cost_breakdown.png", dpi=160)
    plt.close()


def main():
    report_path = Path("ensemble_optimization_report.json")
    out_dir = Path("Results")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = load_report(report_path)
    results = report.get("results", [])

    plot_metric_comparison(report, out_dir)
    plot_model_summary(results, out_dir)
    plot_cost_breakdown(results, out_dir)

    print("Generated plots:")
    print("- Results/results_metrics_comparison.png")
    print("- Results/results_model_summary.png")
    if any("business_cost" in r for r in results):
        print("- Results/results_business_cost_breakdown.png")


if __name__ == "__main__":
    main()
