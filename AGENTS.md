# Customer Segmentation and Retention

A reproducible Python study of churn risk modeling and retention-segmentation policy on the IBM Telco churn dataset.

## Build commands

```bash
# Verification commands.
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Platform support

- macOS and Linux; Python 3.12-3.13; CPU-only development and benchmarks.
- Kaggle runs the frozen benchmark on CPU via `scripts/kaggle_run.sh`.

## Key conventions

- The Telco CSV is verified by sha256 and never committed; the Kaggle runner downloads it from `blastchar/telco-customer-churn` into a pinned data dataset.
- Stratified 60/20/20 train/validation/test split, seed 42; every preprocessing step fits on train only.
- Optuna tunes on train CV; calibration and model selection use validation only; thresholds freeze on validation before the first test read.
- `manifest.json` records `card_write_epoch < first_test_read_epoch` (`ordering_ok`).
- Retention costs (FN 780, FP 20) are declared assumptions, not measured costs; reported expected costs inherit that caveat.
- Retention actions are recommendations under stated cost assumptions, not measured savings.

## Rules

- Default branch is `main`; never force-push it or rewrite shared history.
- Ask before adding any dependency or package manager.
- Never fabricate data, benchmarks, logs or results.
- Done means the relevant tests actually ran and passed; never skip or mock them to finish.
- No emojis anywhere: code, comments, commits, READMEs, docs.
- No em dashes (the long dash, U+2014) anywhere: code, comments, commits, READMEs, docs. Use a comma, colon, parentheses, or a new sentence instead.
- Keep README in sync with code changes, following its existing template.
- Comments are brief and explain why, only when the code doesn't already say it.
- Keep dataset splits and eval leakage controls intact.

## Commits

- Conventional Commits: `type(scope): summary`. Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`; scope is optional.
- Summary in imperative mood, lowercase, no trailing period, at most 72 characters.
- One logical change per commit; keep formatting-only changes separate.
- Add a body only when the reason is not obvious: explain why, not what, wrapped at 72.
- Breaking changes: `type!:` in the summary plus a `BREAKING CHANGE:` footer.
- Commit only when asked. Never commit secrets, build output, or failing code.
