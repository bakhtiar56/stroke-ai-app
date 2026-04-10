from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from stroke_ai.models.importance import save_permutation_importance


@dataclass(frozen=True)
class Config:
    data_path: Path = Path("data/raw/stroke.csv")
    artifacts_dir: Path = Path("artifacts")
    random_state: int = 42
    test_size: float = 0.25
    top_k: float = 0.10


FEATURES = [
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
]
TARGET = "stroke"


def _load_data(cfg: Config) -> pd.DataFrame:
    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {cfg.data_path}. Place your CSV at data/raw/stroke.csv"
        )
    df = pd.read_csv(cfg.data_path)

    # Normalize missing tokens commonly found in this dataset
    df = df.replace(["N/A", "NA", "nan", "NaN", ""], np.nan)

    # Ensure required columns exist
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Coerce numeric
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["avg_glucose_level"] = pd.to_numeric(df["avg_glucose_level"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # Coerce binarys
    df["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce").fillna(0).astype(int)
    df["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce").fillna(0).astype(int)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

    return df


def _build_pipeline() -> Pipeline:
    numeric = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
    categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    return Pipeline([("preprocess", pre), ("model", clf)])


def _exact_topk_metrics(y_true: np.ndarray, y_score: np.ndarray, top_k: float) -> dict[str, float]:
    n = len(y_score)
    k = int(np.ceil(top_k * n))
    k = max(0, min(k, n))

    order = np.argsort(-y_score, kind="mergesort")
    flagged = np.zeros(n, dtype=bool)
    if k > 0:
        flagged[order[:k]] = True

    # Precision/recall on flagged subset (treat flagged as predicted positive)
    precision = precision_score(y_true, flagged, zero_division=0)
    recall = recall_score(y_true, flagged, zero_division=0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Reference threshold: the score at the k-th rank (min score among flagged)
    threshold_ref = float(y_score[order[k - 1]]) if k > 0 else 1.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "flag_rate_test": float(flagged.mean()) if n else 0.0,
        "threshold_reference": float(threshold_ref),
    }


def main() -> None:
    cfg = Config()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = _load_data(cfg)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # Oversample train only
    ros = RandomOverSampler(random_state=cfg.random_state)
    X_train_os, y_train_os = ros.fit_resample(X_train, y_train)

    model = _build_pipeline()
    model.fit(X_train_os, y_train_os)

    # Predict
    p_test = model.predict_proba(X_test)[:, 1].astype(float)
    p_test = np.nan_to_num(p_test, nan=0.0, posinf=1.0, neginf=0.0)

    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "brier": float(brier_score_loss(y_test, p_test)),
    }
    topk = _exact_topk_metrics(y_test.to_numpy(), p_test, cfg.top_k)
    metrics.update({k: topk[k] for k in ["precision", "recall", "f1", "flag_rate_test"]})

    # Save model
    joblib.dump(model, cfg.artifacts_dir / "model.joblib")

    # Save policy
    (cfg.artifacts_dir / "decision_policy.json").write_text(
        json.dumps({"policy": "top_k", "top_k": cfg.top_k, "method": "exact_rank"}, indent=2),
        encoding="utf-8",
    )

    # Save model info
    model_info = {
        "model_version": "0.1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "decision_policy": {"policy": "top_k", "top_k": cfg.top_k, "method": "exact_rank"},
        "threshold_reference": topk["threshold_reference"],
        "metrics": metrics,
        "features": FEATURES,
        "imbalance_strategy": "random_oversample_train_only",
    }
    (cfg.artifacts_dir / "model_info.json").write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )

    # Save permutation importance (global, test set)
    save_permutation_importance(
        model=model,
        X=X_test,
        y=y_test,
        out_path=cfg.artifacts_dir / "feature_importance.json",
        scoring="roc_auc",
        n_repeats=5,
        top_n=15,
    )

    print("Training complete.")
    print(f"Artifacts written to: {cfg.artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()