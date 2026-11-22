"""sklearn Pipelines fit on the training split only.

One shared feature pipeline serves every learner: numerics get median
imputation plus standard scaling, categoricals get one-hot encoding with
unknown handling. Fitting happens exactly once on train; validation and
test are only transformed.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data.telco import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_feature_pipeline() -> ColumnTransformer:
    """ColumnTransformer over the Telco feature columns."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def feature_names(pipe: ColumnTransformer) -> list[str]:
    """Transformed column names in output order."""
    return [str(c) for c in pipe.get_feature_names_out()]
