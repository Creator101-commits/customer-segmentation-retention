"""Tests for Telco data loading, cleaning, and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.data.telco import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TELCO_CSV_SHA256,
    clean_telco,
    load_telco,
    load_telco_csv,
    stratified_split,
)

COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def _row(i: int, churn: str = "No", tenure: int = 10, total: str = "100") -> list:
    return [
        f"id{i}", "Male", 0, "Yes", "No", tenure, "Yes", "No",
        "DSL", "No", "Yes", "No", "No", "No", "No", "Month-to-month",
        "Yes", "Mailed check", 50.0, total, churn,
    ]


def _write_csv(path: Path, n: int = 300, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        churn = "Yes" if rng.random() < 0.27 else "No"
        tenure = int(rng.integers(0, 72))
        total = "" if (i % 97 == 0) else str(29.85 * tenure)
        rows.append(_row(i, churn=churn, tenure=tenure, total=total))
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def test_sha256_constant_matches_committed_csv() -> None:
    # The pinned digest must equal the CSV that was in git history.
    assert TELCO_CSV_SHA256 == (
        "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
    )


def test_load_rejects_wrong_sha(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_telco_csv(csv)


def test_clean_imputes_total_charges_blanks() -> None:
    df = pd.DataFrame(
        [_row(0, total=""), _row(1, total="55.5")], columns=COLUMNS
    )
    out, blanks, zero_blanks = clean_telco(df)
    assert blanks == 1
    assert zero_blanks == 0  # tenure=10 in fixture, not 0
    assert out["TotalCharges"].isna().sum() == 0
    assert out.loc[0, "TotalCharges"] == 0.0
    assert out["Churn"].dtype == np.int64


def test_split_sizes_and_stratification() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.26).astype(int)
    train, val, test = stratified_split(y)
    assert len(train) == 1200 and len(val) == 400 and len(test) == 400
    for idx in (train, val, test):
        assert abs(y[idx].mean() - y.mean()) < 0.02
    train2, val2, test2 = stratified_split(y)
    np.testing.assert_array_equal(train, train2)


def test_load_telco_no_verify(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "t.csv")
    data = load_telco(csv, verify=False)
    assert data.total_charges_blanks > 0
    assert data.blanks_with_zero_tenure >= 0
    assert len(data.train_idx) + len(data.val_idx) + len(data.test_idx) == len(data.df)
    assert set(FEATURES) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
