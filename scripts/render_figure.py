"""Render image.png from the committed results/ artifacts only.

Panels: (a) test ROC-AUC/PR-AUC per learner with bootstrap CIs, (b) frozen
thresholds on the selected model, (c) observed churn rate vs mean predicted
risk per segment, (d) tenure x MonthlyCharges segment grid, (e) SHAP global
importance. Requires Matplotlib; run with
`uv run --with matplotlib python scripts/render_figure.py`.
Set CHURN_RESULTS_DIR to render from a different results directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("CHURN_RESULTS_DIR", ROOT / "results"))
OUT = ROOT / "image.png"

INK = "#1F2A37"
BLUE = "#3B6EA8"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#D1495B"
GRAY = "#9AA5B1"
GRID = "#EEF0F2"
SPINE = "#B8BEC6"
TICK_LABEL = "#4A5361"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": TICK_LABEL,
        "ytick.labelcolor": TICK_LABEL,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

DPI = 150
ORDER = ["majority", "logistic", "xgboost", "lightgbm"]


def _fmt2(x: float, _pos: int) -> str:
    """Up to 2 decimals inside mathtext; large values as integers."""
    if x == 0:
        return "$0$"
    if abs(x) >= 1000:
        return f"${x:.0f}$"
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "$0$"
    return f"${s}$"


CATEGORICAL: list[tuple[plt.Axes, str, list[str]]] = []


def style(
    ax: plt.Axes,
    letter: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=11, color=INK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color(SPINE)
        ax.spines[name].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.6,
        color=SPINE,
        labelcolor=TICK_LABEL,
        labelsize=8.5,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def cat(
    ax: plt.Axes,
    axis: str,
    positions: list[int] | range,
    labels: list[str],
    rotation: float = 0,
    ha: str | None = None,
    fontsize: float | None = None,
) -> None:
    """Categorical ticks: FixedLocator + FixedFormatter applied AFTER
    style(), so the numeric FuncFormatter cannot clobber the labels.
    Rotation/alignment/fontsize are set on the label Texts directly."""
    labels = list(labels)
    if axis == "x":
        ax.set_xticks(positions)
        ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_xticklabels()
    else:
        ax.set_yticks(positions)
        ax.yaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_yticklabels()
    for t in texts:
        if rotation:
            t.set_rotation(rotation)
        if ha:
            t.set_ha(ha)
        if fontsize:
            t.set_fontsize(fontsize)
    CATEGORICAL.append((ax, axis, labels))


def verify_categories() -> None:
    bad = []
    for ax, axis, expected in CATEGORICAL:
        got = [
            t.get_text()
            for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())
        ]
        if got != expected:
            bad.append((axis, expected, got))
    CATEGORICAL.clear()
    if bad:
        details = "; ".join(f"{a} expected={e} got={g}" for a, e, g in bad)
        raise RuntimeError(f"categorical tick labels mismatch: {details}")


def _all_texts(fig: plt.Figure) -> list[plt.Text]:
    items = []
    for ax in fig.axes:
        cand = [ax.title, ax.xaxis.label, ax.yaxis.label]
        cand += list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            try:
                ticks = axis._update_ticks()  # noqa: SLF001
            except Exception:
                ticks = []
            for tick in ticks:
                lab = tick.label1
                if lab.get_text():
                    cand.append(lab)
        leg = ax.get_legend()
        if leg is not None:
            cand += list(leg.get_texts())
        for t in cand:
            try:
                if t.get_text() and t.get_visible():
                    items.append(t)
            except Exception:
                pass
    return items


def check_text_overlap(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    boxes = []
    for t in _all_texts(fig):
        bb = t.get_window_extent(ren)
        boxes.append((t.get_text()[:40], bb))
    problems = []
    for i, (s1, b1) in enumerate(boxes):
        if b1.x0 < -1 or b1.y0 < -1 or b1.x1 > W + 1 or b1.y1 > H + 1:
            problems.append(f"OUT-OF-FIGURE {s1!r} bbox={b1}")
        for s2, b2 in boxes[i + 1 :]:
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 1.5 and iy > 1.5:
                inter = ix * iy
                small = min(b1.width * b1.height, b2.width * b2.height)
                if inter > max(8.0, 0.10 * small):
                    problems.append(f"OVERLAP {s1!r} x {s2!r} area={inter:.0f}px2")
    return problems


def check_legend_data(fig: plt.Figure) -> list[str]:
    """Count data markers/vertices falling inside legend boxes."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(ren)
        covered = 0
        total = 0
        for coll in ax.collections:
            offs = coll.get_offsets()
            if len(offs) == 0 or len(offs) > 2000:
                continue
            pts = ax.transData.transform(offs)
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(offs)
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            try:
                pts = ax.transData.transform(line.get_path().vertices)
            except Exception:
                continue
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(pts)
        for patch in ax.patches:
            try:
                pb = patch.get_window_extent(ren)
            except Exception:
                continue
            if pb.x0 < lb.x1 and pb.x1 > lb.x0 and pb.y0 < lb.y1 and pb.y1 > lb.y0:
                covered += 1
            total += 1
        if covered:
            out.append(f"legend covers ~{covered}/{total} data items")
    return out


def check_annotation_data(fig: plt.Figure) -> list[str]:
    """In-axes annotations vs plotted data."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text() or not t.get_visible():
                continue
            tb = t.get_window_extent(ren)
            hits = 0
            for line in ax.lines:
                if line.get_transform() != ax.transData:
                    continue
                pts = ax.transData.transform(line.get_path().vertices)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            for coll in ax.collections:
                offs = coll.get_offsets()
                if len(offs) == 0 or len(offs) > 5000:
                    continue
                pts = ax.transData.transform(offs)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            if hits:
                out.append(f"'{t.get_text()}' overlaps ~{hits} data pts")
    return out


def check_legend_pixels(fig: plt.Figure, path: Path) -> list[str]:
    """Saturated (colored) data pixels showing through legend boxes."""
    if not path.exists():
        return []
    img = plt.imread(path)
    H, W = img.shape[:2]
    rgb = img[..., :3]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    colored = spread > 0.18
    def _collect(o: object, swatches: list) -> None:
        if type(o).__name__ == "DrawingArea":
            swatches.append(o)
            return
        try:
            children = o.get_visible_children()
        except Exception:
            children = []
        for c in children:
            _collect(c, swatches)

    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        x0, x1 = int(bb.x0), min(int(bb.x1) + 1, W)
        y0, y1 = max(0, int(H - bb.y1)), min(H, int(H - bb.y0) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = colored[y0:y1, x0:x1].copy()
        # mask every legend handle swatch (any column) so only data-colored
        # bleed through the legend body counts; swatches live in
        # DrawingArea children of the legend's offset box
        swatches: list = []
        box = getattr(leg, "_legend_handle_box", None)
        if box is not None:
            _collect(box, swatches)
        for sw in swatches:
            try:
                hb = sw.get_window_extent(fig.canvas.get_renderer())
            except Exception:
                continue
            if not all(np.isfinite([hb.x0, hb.x1, hb.y0, hb.y1])):
                continue
            pad = 3
            hx0, hx1 = int(hb.x0) - x0 - pad, int(hb.x1) - x0 + pad
            hy0, hy1 = int(H - hb.y1) - y0 - pad, int(H - hb.y0) - y0 + pad
            rx0, rx1 = max(0, hx0), min(region.shape[1], hx1)
            ry0, ry1 = max(0, hy0), min(region.shape[0], hy1)
            if rx1 > rx0 and ry1 > ry0:
                region[ry0:ry1, rx0:rx1] = False
        frac = region.mean() if region.size else 0.0
        if frac > 0.015:
            out.append(f"legend body ~{frac:.1%} covered by data color (ax={ax.get_subplotspec()})")
    return out


def finish(fig: plt.Figure, out: Path) -> None:
    """Verify categorical labels, save, then fail on any check violation."""
    fig.canvas.draw()
    fig.canvas.draw()
    verify_categories()
    fig.savefig(out, dpi=DPI, facecolor="white")
    problems: list[str] = []
    problems += check_text_overlap(fig)
    problems += check_legend_data(fig)
    problems += check_annotation_data(fig)
    problems += check_legend_pixels(fig, out)
    plt.close(fig)
    if problems:
        for p in problems[:30]:
            print(f"  CHECK FAIL: {p}", file=sys.stderr)
        sys.exit(f"figure checks failed for {out.name}: {len(problems)} problem(s)")
    print("wrote", out)


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    segments = pd.read_csv(RESULTS / "segments.csv")
    shap = pd.read_csv(RESULTS / "shap_importance.csv")
    sel = metrics["selected_model"]
    names = [n for n in ORDER if n in metrics["models"]]

    fig, axes = plt.subplots(2, 3, figsize=(13.9, 6.3), dpi=DPI, layout="constrained")
    (a1, a2, a3), (a4, a5, _a6) = axes

    # (a) Test discrimination per learner with bootstrap CIs.
    x = np.arange(len(names))
    for key, color in (("roc_auc", BLUE), ("pr_auc", TEAL)):
        off = -0.18 if key == "roc_auc" else 0.18
        means, los, his = [], [], []
        for n in names:
            ci = metrics["models"][n]["test"]["ci"][key]
            m, lo, hi = ci["mean"], ci["ci_low"], ci["ci_high"]
            means.append(m)
            los.append(m - lo)
            his.append(hi - m)
        a1.errorbar(
            x + off,
            means,
            yerr=[los, his],
            fmt="o",
            capsize=3,
            ms=5,
            label=key.upper().replace("_", "-"),
            color=color,
        )
    a1.set_ylim(0.0, 1.0)
    style(a1, "a", "Test discrimination (95% bootstrap CI)",
          ylabel="score", xlabel="learner")
    cat(a1, "x", x, names)
    a1.legend(loc="lower right")

    # (b) Frozen thresholds on the selected model.
    pt = metrics["models"][sel]["test"]["point"]
    groups = ["cost_optimal", "max_precision_recall_0.80"]
    glabels = ["cost-optimal", "max precision\nrecall $\\geq$ 0.80"]
    width = 0.26
    xpos = np.arange(len(groups))
    for j, (key, color) in enumerate(
        [("precision", BLUE), ("recall", TEAL), ("f1", CORAL)]
    ):
        vals = [pt[f"{g}_{key}"] for g in groups]
        bars = a2.bar(
            xpos + (j - 1) * width, vals, width, label=key, color=color
        )
        for bar, val in zip(bars, vals, strict=True):
            a2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}",
                ha="center",
                fontsize=7.5,
                color=INK,
            )
    a2.set_ylim(0.0, 1.35)
    style(a2, "b", f"Frozen thresholds on test ({sel})",
          ylabel="score", xlabel="threshold policy")
    cat(a2, "x", xpos, glabels)
    a2.legend(loc="upper right")

    # (c) Observed churn rate vs mean predicted risk per segment.
    seg = metrics["segments"]
    risks = np.array([s["mean_predicted_risk"] for s in seg])
    churn = np.array([s["churn_rate"] for s in seg])
    counts = np.array([s["n"] for s in seg], dtype=float)
    sizes = 24 + 320 * counts / counts.max()
    a3.scatter(
        risks, churn, s=sizes, color=BLUE, alpha=0.75,
        edgecolors="white", linewidths=0.6, zorder=3,
    )
    lim = max(0.55, float(max(risks.max(), churn.max())) * 1.08)
    a3.plot([0, lim], [0, lim], color=GRAY, lw=1.0, ls="--", zorder=2)
    a3.set_xlim(0, lim)
    a3.set_ylim(0, lim)
    a3.set_xlabel("mean predicted risk")
    a3.set_ylabel("observed churn rate")
    style(a3, "c", "Calibration by segment (area $\\propto$ n)", grid=False)
    a3.text(
        lim * 0.97, lim * 0.05, "perfect calibration",
        ha="right", va="bottom", fontsize=7.5, color=TICK_LABEL,
        rotation=38,
    )

    # (d) Segment grid: churn rate heatmap with mean risk and n annotated.
    segdf = segments.copy()
    segdf[["tenure_band", "charge_band"]] = segdf["segment"].str.split(
        "__", expand=True
    )
    band_order = {"low": 0, "mid": 1, "high": 2}
    tenure_lv = sorted(segdf["tenure_band"].unique(), key=band_order.get)
    charge_lv = sorted(segdf["charge_band"].unique(), key=band_order.get)
    grid = np.full((len(tenure_lv), len(charge_lv)), np.nan)
    risk = np.full_like(grid, np.nan, dtype=float)
    counts_g = np.zeros_like(grid, dtype=int)
    for _, row in segdf.iterrows():
        i = tenure_lv.index(row["tenure_band"])
        j = charge_lv.index(row["charge_band"])
        grid[i, j] = row["churn_rate"]
        risk[i, j] = row["mean_predicted_risk"]
        counts_g[i, j] = row["n"]
    vmax = max(0.5, float(np.nanmax(grid)))
    im = a4.imshow(grid, cmap="YlOrRd", vmin=0.0, vmax=vmax)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            a4.text(
                j,
                i,
                f"{grid[i, j]:.2f}\n({risk[i, j]:.2f})\nn={counts_g[i, j]}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if grid[i, j] > 0.6 * vmax else INK,
            )
    style(a4, "d", "Test churn rate (mean risk, n)",
          xlabel="MonthlyCharges tertile", ylabel="tenure tertile",
          grid=False)
    cat(a4, "x", range(len(charge_lv)), charge_lv, fontsize=8)
    cat(a4, "y", range(len(tenure_lv)),
        [f"tenure {v}" for v in tenure_lv], fontsize=8)
    for sp in a4.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=a4, fraction=0.046, pad=0.02)

    # (e) SHAP top features.
    top = shap.head(10).iloc[::-1]
    a5.barh(
        top["feature"], top["mean_abs_shap"], color=BLUE, height=0.62
    )
    a5.set_xlabel("mean $|$SHAP$|$")
    style(a5, "e", f"SHAP importance ({sel}, test)")
    cat(a5, "y", range(len(top)), list(top["feature"]), fontsize=8)

    finish(fig, OUT)


if __name__ == "__main__":
    main()
