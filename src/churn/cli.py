"""CLI for the churn benchmark."""

from __future__ import annotations

from pathlib import Path

import typer

from churn.experimentation.benchmark import run_benchmark

app = typer.Typer(help="Churn modeling and retention-segmentation benchmark.")


@app.command()
def benchmark(
    data: Path = typer.Option(..., "--data", help="Path to the Telco churn CSV."),  # noqa: B008
    out: Path = typer.Option(Path("benchmark"), "--out", help="Output directory."),  # noqa: B008
    trials: int = typer.Option(100, "--trials", help="Optuna trials per learner."),  # noqa: B008
    boot: int = typer.Option(2000, "--boot", help="Bootstrap resamples."),  # noqa: B008
    smoke: bool = typer.Option(False, "--smoke", help="Reduced diagnostic run."),  # noqa: B008
    no_verify: bool = typer.Option(  # noqa: B008
        False, "--no-verify", help="Skip the source CSV sha256 check."
    ),
) -> None:
    """Run the frozen train/val/test churn benchmark."""
    if smoke:
        trials = min(trials, 8)
        boot = min(boot, 100)
    result = run_benchmark(
        data,
        out,
        n_trials=trials,
        n_boot=boot,
        smoke=smoke,
        verify=not no_verify,
    )
    manifest = result["manifest"]
    typer.echo(f"selected_model: {result['metrics']['selected_model']}")
    typer.echo(f"ordering_ok: {manifest['ordering_ok']}")
    typer.echo(f"artifacts: {out}")


if __name__ == "__main__":
    app()
