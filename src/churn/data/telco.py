"""IBM Telco churn dataset: load, verify, clean, and split.

Source: Kaggle ``blastchar/telco-customer-churn`` (7,043 rows). The raw
CSV is not committed; ``TELCO_CSV_SHA256`` pins the expected digest so any
mirror or re-download is checked byte-for-byte.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TELCO_CSV_SHA256 = (
    "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
)
TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class TelcoData:
    """Cleaned frame plus split indices and provenance notes."""

    df: pd.DataFrame
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    total_charges_blanks: int
    blanks_with_zero_tenure: int
    source_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_telco_csv(path: Path, *, verify: bool = True) -> pd.DataFrame:
    """Load the raw CSV, verify its sha256, and return the frame."""
    if verify:
        actual = _sha256_file(path)
        if actual != TELCO_CSV_SHA256:
            raise ValueError(
                f"sha256 mismatch for {path}: {actual} != {TELCO_CSV_SHA256}"
            )
    df = pd.read_csv(path)
    return df


def clean_telco(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Parse TotalCharges, impute its blanks to 0, return (df, blanks, zero_tenure_blanks)."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    blanks = int(out["TotalCharges"].isna().sum())
    zero_tenure_blanks = int(
        (out["TotalCharges"].isna() & (out["tenure"] == 0)).sum()
    )
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)
    out[TARGET] = (out[TARGET] == "Yes").astype(int)
    return out, blanks, zero_tenure_blanks


def stratified_split(
    y: np.ndarray, seed: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 60/20/20 stratified split; returns train/val/test indices."""
    idx = np.arange(len(y))
    train_val, test = train_test_split(
        idx, test_size=TEST_FRAC, random_state=seed, stratify=y
    )
    rel_val = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=y[train_val]
    )
    return train, val, test


def load_telco(
    path: Path, *, verify: bool = True, seed: int = RANDOM_STATE
) -> TelcoData:
    """Full pipeline: verify, clean, encode target, split."""
    raw = load_telco_csv(path, verify=verify)
    df, blanks, zero_blanks = clean_telco(raw)
    y = df[TARGET].to_numpy(dtype=np.int64)
    train_idx, val_idx, test_idx = stratified_split(y, seed=seed)
    return TelcoData(
        df=df,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        total_charges_blanks=blanks,
        blanks_with_zero_tenure=zero_blanks,
        source_sha256=TELCO_CSV_SHA256,
    )


def data_manifest(data: TelcoData, path: Path) -> dict[str, Any]:
    """Provenance record for manifest.json."""
    df = data.df
    return {
        "source": "kaggle blastchar/telco-customer-churn",
        "source_csv_sha256": data.source_sha256,
        "n_rows": int(len(df)),
        "n_features": len(FEATURES),
        "target": TARGET,
        "target_positive_rate": float(data.y.mean()),
        "total_charges_blanks_imputed_to_zero": data.total_charges_blanks,
        "blanks_with_zero_tenure": data.blanks_with_zero_tenure,
        "split": {
            "scheme": "stratified 60/20/20",
            "seed": RANDOM_STATE,
            "train": int(len(data.train_idx)),
            "validation": int(len(data.val_idx)),
            "test": int(len(data.test_idx)),
            "train_churn_rate": float(
                data.y[data.train_idx].mean()
            ),
            "val_churn_rate": float(data.y[data.val_idx].mean()),
            "test_churn_rate": float(data.y[data.test_idx].mean()),
        },
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "path": str(path),
    }
