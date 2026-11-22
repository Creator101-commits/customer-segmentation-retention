"""Kaggle entrypoint for the Telco churn benchmark (CPU).

Installs ``churn`` from the pinned source dataset uploaded by
``scripts/kaggle_run.sh``, locates the sha256-verified Telco CSV in the
data dataset, runs the frozen benchmark, and writes ``benchmark/``
artifacts including ``manifest.json`` with the git SHA. ``CHURN_SMOKE=1``
reduces Optuna trials and bootstrap resamples for a diagnostic run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BENCH = WORK / "benchmark"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "churn").is_dir():
            return candidate.parent
    return None


def _install_package() -> Path:
    proj_root = _find_project_root()
    print(f"project root: {proj_root}", flush=True)
    if proj_root is None:
        raise FileNotFoundError(
            "No installable project found under /kaggle/input "
            "(expected pyproject.toml + src/churn)."
        )
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", str(proj)])
    return proj


def _git_sha(proj: Path) -> str:
    info = proj / "run_info.json"
    if info.exists():
        return str(json.loads(info.read_text()).get("git_sha", "unknown"))
    return "unknown"


def _find_csv() -> Path:
    matches = [
        p
        for p in KAGGLE_INPUT.rglob("*.csv")
        if "Telco" in p.name or "telco" in p.name.lower()
    ]
    if not matches:
        matches = list(KAGGLE_INPUT.rglob("*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected 1 CSV under /kaggle/input, found {matches}")
    return matches[0]


def main() -> None:
    proj = _install_package()
    sha = _git_sha(proj)
    print(f"git_sha={sha}", flush=True)

    from churn.data.telco import load_telco_csv  # noqa: PLC0415
    from churn.experimentation.benchmark import run_benchmark  # noqa: PLC0415

    csv_path = _find_csv()
    print(f"telco csv: {csv_path}", flush=True)
    # load_telco_csv verifies sha256 against the committed constant.
    load_telco_csv(csv_path)

    smoke = bool(os.environ.get("CHURN_SMOKE"))
    if smoke:
        print("SMOKE MODE: trials=8 boot=100", flush=True)

    BENCH.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(
        csv_path,
        BENCH,
        n_trials=8 if smoke else 100,
        n_boot=100 if smoke else 2000,
        git_sha=sha,
        smoke=smoke,
    )

    metrics = result["metrics"]
    manifest = result["manifest"]
    print("=== FROZEN ===", flush=True)
    print(json.dumps({"selected_model": metrics["selected_model"],
                      "ordering_ok": manifest["ordering_ok"]}, indent=2), flush=True)
    sel = metrics["models"][metrics["selected_model"]]["test"]["point"]
    for key in ("roc_auc", "pr_auc", "log_loss", "brier", "ece"):
        print(f"{key}: {sel.get(key)}", flush=True)


if __name__ == "__main__":
    main()
