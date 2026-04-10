from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_VERSION = "0.1.0"

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
MODEL_INFO_PATH = ARTIFACTS_DIR / "model_info.json"
POLICY_PATH = ARTIFACTS_DIR / "decision_policy.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"

app = FastAPI(title="Stroke AI", version=APP_VERSION)

# Lazy-loaded globals
_MODEL = None
_MODEL_INFO: dict[str, Any] | None = None
_POLICY: dict[str, Any] | None = None
_FEATURE_IMPORTANCE: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_artifacts() -> None:
    global _MODEL, _MODEL_INFO, _POLICY, _FEATURE_IMPORTANCE
    if _MODEL is not None:
        return

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Missing model artifact: {MODEL_PATH}. Run training to create artifacts.",
        )

    # Load (this is the potentially slow step; do it once)
    _MODEL = joblib.load(MODEL_PATH)
    _MODEL_INFO = _load_json(MODEL_INFO_PATH)
    _POLICY = _load_json(POLICY_PATH)
    _FEATURE_IMPORTANCE = _load_json(FEATURE_IMPORTANCE_PATH)


REQUIRED_COLUMNS = [
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


def _threshold_reference() -> float:
    mi = _MODEL_INFO or {}
    return float(mi.get("threshold_reference", 0.0677837550652022))


def _risk_band(p: float) -> str:
    t = _threshold_reference()
    if p < 0.5 * t:
        return "low"
    if p < t:
        return "medium"
    return "high"


def _coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required fields: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.replace(["N/A", "NA", "nan", "NaN", ""], np.nan)

    for c in ["age", "avg_glucose_level", "bmi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["hypertension", "heart_disease"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


class PredictRequest(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float | None = None
    smoking_status: str


class ScreenRequest(BaseModel):
    top_k: float = Field(0.10, ge=0.0, le=1.0)
    records: list[dict[str, Any]]


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    # Try loading so healthz tells you what's wrong instead of timing out later
    try:
        _load_artifacts()
        loaded = True
        err = ""
    except HTTPException as e:
        loaded = False
        err = str(e.detail)

    return {
        "ok": True,
        "version": APP_VERSION,
        "model_loaded": loaded,
        "artifacts_dir": str(ARTIFACTS_DIR.resolve()),
        "model_path": str(MODEL_PATH.resolve()),
        "error": err,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    _load_artifacts()
    return _MODEL_INFO or {}


@app.get("/policy")
def policy() -> dict[str, Any]:
    _load_artifacts()
    return _POLICY or {}


@app.get("/feature-importance")
def feature_importance() -> dict[str, Any]:
    _load_artifacts()
    return _FEATURE_IMPORTANCE or {}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    _load_artifacts()

    df = pd.DataFrame([req.model_dump()])
    df = _coerce_df(df)

    risks = _MODEL.predict_proba(df)[:, 1].astype(float)
    risks = np.nan_to_num(risks, nan=0.0, posinf=1.0, neginf=0.0)
    risk = float(risks[0])

    return {
        "risk_score": risk,
        "risk_band": _risk_band(risk),
        "screening_threshold_reference": _threshold_reference(),
        "top_factors": (_FEATURE_IMPORTANCE or {}).get("top_factors", []),
    }


@app.post("/screen")
def screen(req: ScreenRequest) -> dict[str, Any]:
    _load_artifacts()

    if not req.records:
        raise HTTPException(status_code=422, detail="records must be non-empty")

    df = _coerce_df(pd.DataFrame(req.records))

    risks = _MODEL.predict_proba(df)[:, 1].astype(float)
    risks = np.nan_to_num(risks, nan=0.0, posinf=1.0, neginf=0.0)
    # HARD CHECK (prevents JSON crash no matter what)
    if not np.all(np.isfinite(risks)):
        bad = np.where(~np.isfinite(risks))[0].tolist()
        raise HTTPException(
            status_code=422,
            detail=f"Non-finite risk scores after sanitization for row indices: {bad[:20]}",
        )

    n = len(risks)
    k = int(np.ceil(req.top_k * n))
    k = max(0, min(k, n))

    order = np.argsort(-risks, kind="mergesort")
    flagged = np.zeros(n, dtype=bool)
    if k > 0:
        flagged[order[:k]] = True

    results = [
        {
            "index": i,
            "risk_score": float(risks[i]),
            "risk_band": _risk_band(float(risks[i])),
            "flagged": bool(flagged[i]),
        }
        for i in range(n)
    ]

    return {
        "policy": {
            "policy": "top_k",
            "top_k": float(req.top_k),
            "method": "exact_rank",
            "k": int(k),
            "screening_threshold_reference": _threshold_reference(),
        },
        "total": int(n),
        "flagged_count": int(flagged.sum()),
        "results": results,
    }