from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from stroke_ai.data.load import load_stroke_csv
from stroke_ai.data.validate import CATEGORICAL_COLS, NUMERIC_COLS, split_xy, validate_dataframe

ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    brier: float
    threshold: float
    recall: float
    precision: float
    f1: float
    confusion_matrix: list[list[int]]
    positive_rate_train: float
    positive_rate_val: float
    positive_rate_test: float
    trained_at: str


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_pipeline(random_state: int) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def _pick_threshold(y_val: np.ndarray, proba_val: np.ndarray, min_recall: float = 0.85) -> float:
    # Candidate thresholds from unique probabilities
    thresholds = np.unique(proba_val)
    # Make sure 0/1 extremes are considered
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))

    best = None  # (meets_recall, precision, -threshold, threshold)
    for t in thresholds:
        pred = (proba_val >= t).astype(int)
        rec = recall_score(y_val, pred, zero_division=0)
        prec = precision_score(y_val, pred, zero_division=0)

        meets = rec >= min_recall
        key = (1 if meets else 0, prec, -t, t)
        if best is None or key > best[0]:
            best = (key, t)

    chosen = float(best[1])

    # If nothing meets recall, fall back to best F1
    pred = (proba_val >= chosen).astype(int)
    if recall_score(y_val, pred, zero_division=0) < min_recall:
        best_f1 = -1.0
        best_t = 0.5
        for t in thresholds:
            pred = (proba_val >= t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        chosen = best_t

    return chosen


def _risk_band(p: float) -> str:
    # Simple bands for UI. (Not clinical.)
    if p < 0.10:
        return "low"
    if p < 0.30:
        return "medium"
    return "high"


def _plot_roc(y_true: np.ndarray, proba: np.ndarray, outpath: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return float(auc)


def _plot_pr(y_true: np.ndarray, proba: np.ndarray, outpath: Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (test)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return float(ap)


def _plot_calibration(y_true: np.ndarray, proba: np.ndarray, outpath: Path, n_bins: int = 10) -> float:
    # Lightweight reliability plot without extra deps
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_conf = []
    bin_acc = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_conf.append(float(proba[mask].mean()))
        bin_acc.append(float(y_true[mask].mean()))

    brier = brier_score_loss(y_true, proba)

    plt.figure()
    plt.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")
    plt.plot(bin_conf, bin_acc, marker="o", label=f"Brier={brier:.3f}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration (reliability) curve (test)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return float(brier)


def main() -> None:
    random_state = 42
    _ensure_dirs()

    df = load_stroke_csv()
    report = validate_dataframe(df)
    print(f"Loaded rows: {report.n_rows}")

    X, y = split_xy(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    pipe = _build_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)

    proba_val = pipe.predict_proba(X_val)[:, 1]
    threshold = _pick_threshold(y_val.to_numpy(), proba_val, min_recall=0.85)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)

    roc_auc = _plot_roc(y_test.to_numpy(), proba_test, PLOTS_DIR / "roc_curve.png")
    pr_auc = _plot_pr(y_test.to_numpy(), proba_test, PLOTS_DIR / "pr_curve.png")
    brier = _plot_calibration(y_test.to_numpy(), proba_test, PLOTS_DIR / "calibration.png")

    cm = confusion_matrix(y_test, pred_test).tolist()
    rec = float(recall_score(y_test, pred_test, zero_division=0))
    prec = float(precision_score(y_test, pred_test, zero_division=0))
    f1 = float(f1_score(y_test, pred_test, zero_division=0))

    trained_at = datetime.now(timezone.utc).isoformat()
    metrics = Metrics(
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        brier=float(brier),
        threshold=float(threshold),
        recall=rec,
        precision=prec,
        f1=f1,
        confusion_matrix=cm,
        positive_rate_train=float(y_train.mean()),
        positive_rate_val=float(y_val.mean()),
        positive_rate_test=float(y_test.mean()),
        trained_at=trained_at,
    )

    # Permutation importance on validation set
    perm = permutation_importance(
        pipe, X_val, y_val, n_repeats=15, random_state=random_state, scoring="roc_auc", n_jobs=-1
    )

    # Get transformed feature names from the preprocessing step
    preprocess = pipe.named_steps["preprocess"]
    feature_names = []
    # numeric features
    feature_names.extend(NUMERIC_COLS)
    # categorical OHE feature names
    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    feature_names.extend(cat_feature_names)

    importances = perm.importances_mean
    fi = sorted(
        [{"feature": str(f), "importance": float(i)} for f, i in zip(feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # Normalize for API display (optional but nice)
    topn = fi[:15]
    denom = sum(abs(x["importance"]) for x in topn) or 1.0
    for x in topn:
        x["importance"] = float(abs(x["importance"]) / denom)

    # Save artifacts
    joblib.dump(pipe, ARTIFACTS_DIR / "model.joblib")

    (ARTIFACTS_DIR / "threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2))
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2))
    (ARTIFACTS_DIR / "feature_importance.json").write_text(json.dumps({"top_factors": topn}, indent=2))

    # Also save a tiny model-info summary for convenience
    model_info = {
        "model_version": "0.1.0",
        "trained_at": trained_at,
        "threshold": float(threshold),
        "risk_band_note": "Bands are non-clinical demo buckets: <0.10 low, <0.30 medium, else high.",
        "metrics": {
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
            "brier": metrics.brier,
            "recall": metrics.recall,
            "precision": metrics.precision,
            "f1": metrics.f1,
        },
        "features": list(X.columns),
    }
    (ARTIFACTS_DIR / "model_info.json").write_text(json.dumps(model_info, indent=2))

    print("Saved artifacts to artifacts/:")
    print("- model.joblib")
    print("- threshold.json")
    print("- metrics.json")
    print("- feature_importance.json")
    print("- model_info.json")
    print("- plots/*.png")
    print(f"Chosen threshold={threshold:.4f} | test recall={rec:.3f} precision={prec:.3f} f1={f1:.3f}")
    print(f"Example risk bands: low/medium/high; (demo) e.g. p=0.12 -> {_risk_band(0.12)}")


if __name__ == "__main__":
    main()