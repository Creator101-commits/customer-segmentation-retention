"""Learner zoo: majority baseline, logistic, XGBoost, LightGBM.

Each tree/linear learner is tuned with Optuna TPE (100 trials, 5-fold
stratified CV on TRAIN, ROC-AUC objective, seed 42), refit on all of
train, then calibrated sigmoid-vs-isotonic on validation by log loss.
Model selection across learners happens on validation PR-AUC only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from churn.modeling.preprocessing import build_feature_pipeline

SEED = 42
N_TRIALS = 100
N_FOLDS = 5
CALIBRATION_METHODS = ("sigmoid", "isotonic")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_roc_auc(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
    """Mean 5-fold stratified CV ROC-AUC on the training matrix."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    probs = cross_val_predict(
        estimator, X, y, cv=cv, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(y, probs))


def _tune(
    make_estimator: Callable[[dict[str, Any]], BaseEstimator],
    space: Callable[[optuna.Trial], dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Optuna TPE search; returns the best estimator and its params."""

    def objective(trial: optuna.Trial) -> float:
        return _cv_roc_auc(make_estimator(space(trial)), X, y)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = dict(study.best_params)
    return make_estimator(best_params), best_params


def _logreg_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": float(trial.suggest_float("C", 1e-3, 1e2, log=True)),
        "max_iter": 1000,
    }


def _xgb_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": int(trial.suggest_int("max_depth", 3, 7)),
        "learning_rate": float(
            trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        ),
        "n_estimators": int(trial.suggest_int("n_estimators", 50, 500)),
        "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
        "colsample_bytree": float(
            trial.suggest_float("colsample_bytree", 0.6, 1.0)
        ),
        "min_child_weight": float(
            trial.suggest_float("min_child_weight", 1.0, 10.0)
        ),
        "reg_lambda": float(
            trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        ),
        "random_state": SEED,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }


def _lgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "num_leaves": int(trial.suggest_int("num_leaves", 15, 127)),
        "learning_rate": float(
            trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        ),
        "n_estimators": int(trial.suggest_int("n_estimators", 50, 1000)),
        "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
        "colsample_bytree": float(
            trial.suggest_float("colsample_bytree", 0.6, 1.0)
        ),
        "min_child_samples": int(trial.suggest_int("min_child_samples", 5, 50)),
        "reg_lambda": float(
            trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        ),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }


def _xgb_estimator(params: dict[str, Any]) -> BaseEstimator:
    from xgboost import XGBClassifier  # noqa: PLC0415

    return XGBClassifier(**params)


def _lgbm_estimator(params: dict[str, Any]) -> BaseEstimator:
    from lightgbm import LGBMClassifier  # noqa: PLC0415

    return LGBMClassifier(**params)


LEARNER_NAMES = ("logistic", "xgboost", "lightgbm")

LEARNER_SPECS: dict[
    str,
    tuple[Callable[[dict[str, Any]], BaseEstimator], Callable[[optuna.Trial], dict[str, Any]]],
] = {
    "logistic": (lambda p: LogisticRegression(**p), _logreg_space),
    "xgboost": (_xgb_estimator, _xgb_space),
    "lightgbm": (_lgbm_estimator, _lgbm_space),
}

type Calibrator = IsotonicRegression | LogisticRegression


@dataclass
class FittedLearner:
    """A tuned, calibrated learner ready for probability prediction."""

    name: str
    pipeline: Pipeline
    calibrator: Calibrator | None = None
    calibration_method: str | None = None
    val_logloss_by_method: dict[str, float] = field(default_factory=dict)
    tuned_params: dict[str, Any] = field(default_factory=dict)
    is_baseline: bool = False

    def predict_proba(self, X_raw: Any) -> np.ndarray:
        probs = np.asarray(self.pipeline.predict_proba(X_raw))[:, 1]
        cal = self.calibrator
        if cal is None:
            return probs
        if isinstance(cal, LogisticRegression):
            return np.asarray(
                cal.predict_proba(_logit(probs).reshape(-1, 1))[:, 1]
            )
        return np.asarray(cal.predict(probs))


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(clipped / (1 - clipped)))


def _fit_calibrator(
    method: str, probs_val: np.ndarray, y_val: np.ndarray
) -> Calibrator:
    if method == "sigmoid":
        cal = LogisticRegression(max_iter=1000)
        cal.fit(_logit(probs_val).reshape(-1, 1), y_val)
        return cal
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_val, y_val)
    return iso


def fit_baseline(X_train_raw: Any, y_train: np.ndarray) -> FittedLearner:
    """Majority-class DummyClassifier wrapped in the feature pipeline."""
    pipe = Pipeline(
        [
            ("features", build_feature_pipeline()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    pipe.fit(X_train_raw, y_train)
    return FittedLearner(
        name="majority", pipeline=pipe, is_baseline=True
    )


def fit_learner(
    name: str,
    X_train_raw: Any,
    y_train: np.ndarray,
    X_val_raw: Any,
    y_val: np.ndarray,
    *,
    n_trials: int = N_TRIALS,
) -> FittedLearner:
    """Tune on train CV, refit on train, calibrate on validation."""
    make_est, space = LEARNER_SPECS[name]
    feat = build_feature_pipeline()
    X_train_t = feat.fit_transform(X_train_raw)

    tuned_est, best_params = _tune(
        make_est, space, X_train_t, y_train, n_trials
    )
    base = Pipeline([("features", feat), ("model", tuned_est)])
    base.fit(X_train_raw, y_train)

    probs_val = base.predict_proba(X_val_raw)[:, 1]
    scores: dict[str, float] = {}
    calibrators: dict[str, Calibrator] = {}
    for method in CALIBRATION_METHODS:
        cal = _fit_calibrator(method, probs_val, y_val)
        if isinstance(cal, LogisticRegression):
            cal_probs = cal.predict_proba(_logit(probs_val).reshape(-1, 1))[:, 1]
        else:
            cal_probs = cal.predict(probs_val)
        scores[method] = float(log_loss(y_val, cal_probs))
        calibrators[method] = cal
    best_method = min(scores, key=lambda m: scores[m])
    return FittedLearner(
        name=name,
        pipeline=base,
        calibrator=calibrators[best_method],
        calibration_method=best_method,
        val_logloss_by_method=scores,
        tuned_params=best_params,
    )
