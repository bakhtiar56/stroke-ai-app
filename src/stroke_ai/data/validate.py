from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

TARGET_COL = "stroke"
DROP_COLS = ["id"]

NUMERIC_COLS = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CATEGORICAL_COLS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

ALLOWED_CATEGORIES = {
    "gender": {"Male", "Female", "Other"},
    "ever_married": {"Yes", "No"},
    "work_type": {"Private", "Self-employed", "Govt_job", "children", "Never_worked"},
    "Residence_type": {"Urban", "Rural"},
    "smoking_status": {"formerly smoked", "never smoked", "smokes", "Unknown"},
}


@dataclass(frozen=True)
class ValidationReport:
    n_rows: int
    missingness_rate: dict[str, float]


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")


def _validate_target(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'.")
    bad = set(df[TARGET_COL].dropna().unique()) - {0, 1}
    if bad:
        raise ValueError(f"Target column '{TARGET_COL}' must be in {{0,1}}. Found: {bad}")


def _validate_non_negative(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if (s.dropna() < 0).any():
            raise ValueError(f"Column '{c}' contains negative values which are not allowed.")


def _validate_categories(df: pd.DataFrame) -> None:
    # Soft-ish validation: if unexpected categories appear, fail fast (keeps project crisp).
    for col, allowed in ALLOWED_CATEGORIES.items():
        if col not in df.columns:
            continue
        observed = set(df[col].dropna().astype(str).unique())
        unexpected = observed - allowed
        if unexpected:
            raise ValueError(
                f"Unexpected categories in '{col}': {sorted(unexpected)}. "
                f"Allowed: {sorted(allowed)}"
            )


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    _require_columns(df, REQUIRED_COLUMNS)
    _validate_target(df)
    _validate_non_negative(df, ["age", "avg_glucose_level", "bmi"])
    _validate_categories(df)

    missingness = {c: float(df[c].isna().mean()) for c in df.columns}
    return ValidationReport(n_rows=len(df), missingness_rate=missingness)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL] + DROP_COLS)
    y = df[TARGET_COL].astype(int)
    return X, y