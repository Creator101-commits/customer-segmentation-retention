import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OPTUNA_TRIALS = 80
CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV = 3
COST_FN = 200
COST_FP = 20
MIN_RECALL_CONSTRAINT = 0.80
BASELINE = {
    "roc_auc": 0.8421,
    "f1": 0.6185,
    "accuracy": 0.73,
    "precision": 0.50,
    "recall": 0.82,
}
ENGINEERED_FEATURES = [
    "charges_per_tenure",
    "spend_gap",
    "service_count",
    "service_density",
    "mtm_high_charge",
    "tenure_bucket",
    "no_protection",
    "digital_risk",
    "log_monthly",
    "log_total",
    "fiber_no_security",
    "early_mtm",
    "tenure_x_contract",
]


def sanitize_feature_name(name):
    safe = name
    for old, new in [
        (" ", "_"),
        ("-", "_"),
        ("(", ""),
        (")", ""),
        ("/", "_"),
    ]:
        safe = safe.replace(old, new)
    return safe


def add_engineered_features(df):
    df = df.copy()

    # 1 and 2
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["spend_gap"] = (df["MonthlyCharges"] * df["tenure"]) - df["TotalCharges"]

    # 3
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["service_count"] = sum((df[col] == "Yes").astype(int) for col in service_cols)
    df["service_density"] = df["service_count"] / len(service_cols)

    # 4
    high_charge_cutoff = df["MonthlyCharges"].median()
    df["mtm_high_charge"] = (
        (df["Contract"] == "Month-to-month") & (df["MonthlyCharges"] > high_charge_cutoff)
    ).astype(int)

    # 5
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=[5, 4, 3, 2, 1],
        include_lowest=True,
    ).astype(int)

    # 7
    df["digital_risk"] = (
        (df["PaperlessBilling"] == "Yes") & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    # 8 and 9
    df["log_monthly"] = np.log1p(df["MonthlyCharges"])
    df["log_total"] = np.log1p(df["TotalCharges"])

    # Additional targeted churn-risk interactions
    df["fiber_no_security"] = (
        (df["InternetService"] == "Fiber optic") & (df["OnlineSecurity"] == "No")
    ).astype(int)
    df["early_mtm"] = ((df["Contract"] == "Month-to-month") & (df["tenure"] <= 12)).astype(int)

    return df


def col_or_zeros(df, column_name):
    if column_name in df.columns:
        return df[column_name]
    return pd.Series(0, index=df.index)


def prepare_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = add_engineered_features(df)

    cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    df_model.columns = [sanitize_feature_name(col) for col in df_model.columns]

    # 6
    df_model["no_protection"] = (
        (col_or_zeros(df_model, "OnlineSecurity_No") == 1)
        & (col_or_zeros(df_model, "TechSupport_No") == 1)
    ).astype(int)

    # 10
    df_model["tenure_x_contract"] = df_model["tenure"] * col_or_zeros(
        df_model, "Contract_Month_to_month"
    )

    X = df_model.drop(columns=["Churn"])
    y = df_model["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def compute_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }


def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 250, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()


def find_lowest_cost_threshold(y_true, probs, cost_fn=780, cost_fp=20):
    best_threshold = 0.5
    best_cost = float("inf")
    best_metrics = None

    for threshold in np.arange(0.10, 0.90, 0.01):
        preds = (probs >= threshold).astype(int)
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        total_cost = (fp * cost_fp) + (fn * cost_fn)

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(threshold)
            best_metrics = compute_metrics(y_true, probs, best_threshold)
            best_metrics["fp"] = fp
            best_metrics["fn"] = fn

    best_metrics["business_cost"] = float(best_cost)
    best_metrics["cost_fn"] = cost_fn
    best_metrics["cost_fp"] = cost_fp
    return best_metrics


def find_best_threshold_with_recall_constraint(y_true, probs, min_recall=0.80):
    best_metrics = None
    fallback_metrics = None

    for threshold in np.arange(0.10, 0.90, 0.01):
        metrics = compute_metrics(y_true, probs, threshold)

        if fallback_metrics is None or metrics["recall"] > fallback_metrics["recall"]:
            fallback_metrics = metrics

        if metrics["recall"] >= min_recall:
            if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                best_metrics = metrics

    if best_metrics is None:
        best_metrics = fallback_metrics

    best_metrics["min_recall_constraint"] = float(min_recall)
    return best_metrics


def extract_feature_importance(model, feature_names, engineered_features, top_n=20):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    top_importance = [
        {"feature": feature, "importance": float(value)}
        for feature, value in importances.head(top_n).items()
    ]
    engineered_in_top = [feature for feature in engineered_features if feature in importances.head(top_n).index]
    return top_importance, engineered_in_top


def plot_probability_distribution(probs, recall_threshold):
    plt.figure(figsize=(10, 5))
    plt.hist(probs, bins=50, alpha=0.85, color="#1f77b4", edgecolor="white")
    plt.axvline(
        recall_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Recall-constrained threshold = {recall_threshold:.2f}",
    )
    plt.title("Calibrated Probability Distribution")
    plt.xlabel("Predicted churn probability")
    plt.ylabel("Customer count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prob_distribution.png", dpi=160)
    plt.close()


def evaluate_calibrated_variant(name, estimator, X_train, y_train, y_test):
    calibrated_model = CalibratedClassifierCV(
        estimator=estimator,
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV,
    )
    calibrated_model.fit(X_train, y_train)
    probs = calibrated_model.predict_proba(X_test_global)[:, 1]

    metrics_05 = attach_flags(compute_metrics(y_test, probs, threshold=0.50))
    metrics_05["name"] = f"{name}_threshold_0_50"

    constrained = attach_flags(
        find_best_threshold_with_recall_constraint(
            y_test,
            probs,
            min_recall=MIN_RECALL_CONSTRAINT,
        )
    )
    constrained["name"] = f"{name}_recall_constrained_threshold"

    cost_opt = attach_flags(find_lowest_cost_threshold(y_test, probs, cost_fn=COST_FN, cost_fp=COST_FP))
    cost_opt["name"] = f"{name}_cost_optimized_threshold"

    return {
        "name": name,
        "probs": probs,
        "metrics_05": metrics_05,
        "constrained": constrained,
        "cost_opt": cost_opt,
    }


X_test_global = None


def attach_flags(metrics):
    metrics = metrics.copy()
    metrics["all_non_auc_better"] = (
        metrics["f1"] > BASELINE["f1"]
        and metrics["accuracy"] > BASELINE["accuracy"]
        and metrics["precision"] > BASELINE["precision"]
        and metrics["recall"] > BASELINE["recall"]
    )
    metrics["all_better"] = metrics["all_non_auc_better"] and metrics["roc_auc"] > BASELINE["roc_auc"]
    return metrics


def main():
    global X_test_global
    X_train, X_test, y_train, y_test = prepare_data()
    X_test_global = X_test

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    base_model.fit(X_train, y_train)
    base_probs = base_model.predict_proba(X_test)[:, 1]
    base_metrics = attach_flags(compute_metrics(y_test, base_probs, threshold=0.50))
    base_metrics["name"] = "feature_engineered_baseline_xgb"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=OPTUNA_TRIALS)

    tuned_model = XGBClassifier(
        **study.best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    tuned_model.fit(X_train, y_train)
    top_feature_importance, engineered_features_in_top_20 = extract_feature_importance(
        tuned_model,
        X_train.columns,
        ENGINEERED_FEATURES,
        top_n=20,
    )

    xgb_variant = evaluate_calibrated_variant(
        "calibrated_xgb",
        tuned_model,
        X_train,
        y_train,
        y_test,
    )

    lgbm_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_top_feature_importance, lgbm_engineered_features_in_top_20 = extract_feature_importance(
        lgbm_model,
        X_train.columns,
        ENGINEERED_FEATURES,
        top_n=20,
    )

    lgbm_variant = evaluate_calibrated_variant(
        "calibrated_lgbm",
        lgbm_model,
        X_train,
        y_train,
        y_test,
    )

    selected_model_name = "calibrated_xgb"
    selected_variant = xgb_variant
    lgbm_auc = lgbm_variant["constrained"]["roc_auc"]
    xgb_auc = xgb_variant["constrained"]["roc_auc"]
    if lgbm_auc > xgb_auc and lgbm_auc >= 0.845:
        selected_model_name = "calibrated_lgbm"
        selected_variant = lgbm_variant

    plot_probability_distribution(
        selected_variant["probs"],
        selected_variant["constrained"]["threshold"],
    )

    report = {
        "baseline_reference": BASELINE,
        "optuna_trials": OPTUNA_TRIALS,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_cv": CALIBRATION_CV,
        "cost_settings": {"cost_fn": COST_FN, "cost_fp": COST_FP},
        "recall_constraint": MIN_RECALL_CONSTRAINT,
        "best_optuna_params": study.best_params,
        "best_optuna_cv_auc": float(study.best_value),
        "selected_final_model": selected_model_name,
        "top_20_feature_importance": top_feature_importance,
        "engineered_features": ENGINEERED_FEATURES,
        "engineered_features_in_top_20": engineered_features_in_top_20,
        "lgbm_top_20_feature_importance": lgbm_top_feature_importance,
        "lgbm_engineered_features_in_top_20": lgbm_engineered_features_in_top_20,
        "model_comparison": {
            "xgb_recall_constrained_roc_auc": xgb_auc,
            "lgbm_recall_constrained_roc_auc": lgbm_auc,
        },
        "results": [
            base_metrics,
            xgb_variant["metrics_05"],
            xgb_variant["constrained"],
            xgb_variant["cost_opt"],
            lgbm_variant["metrics_05"],
            lgbm_variant["constrained"],
            lgbm_variant["cost_opt"],
        ],
        "recommended_operating_point": selected_variant["constrained"],
        "notes": [
            "Use recall-constrained threshold as primary campaign operating point.",
            "Do not deploy cost-optimized threshold blindly if precision is too low.",
        ],
    }

    with open("ensemble_optimization_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
