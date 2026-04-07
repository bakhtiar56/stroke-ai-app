from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
THRESH_PATH = ARTIFACTS_DIR / "threshold.json"  # reference/monitoring only
POLICY_PATH = ARTIFACTS_DIR / "decision_policy.json"
FI_PATH = ARTIFACTS_DIR / "feature_importance.json"
MODELINFO_PATH = ARTIFACTS_DIR / "model_info.json"


class PredictRequest(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: Optional[float] = None
    smoking_status: str


class TopFactor(BaseModel):
    feature: str
    importance: float


class PredictResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_band: str
    top_factors: List[TopFactor]
    notes: str


class ScreenRequest(BaseModel):
    records: List[PredictRequest]
    top_k: Optional[float] = Field(default=None, gt=0.0, lt=1.0)  # optional override


class ScreenedRecord(BaseModel):
    index: int
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_band: str
    flagged: bool


class ScreenResponse(BaseModel):
    policy: dict
    flagged_count: int
    total: int
    results: List[ScreenedRecord]


app = FastAPI(title="Stroke AI API (screening demo)")

MODEL = None
MODEL_INFO = {}
TOP_FACTORS: list[dict] = []
THRESHOLD_REFERENCE = None
DECISION_POLICY = {"policy": "top_k", "top_k": 0.10}


def _risk_band(p: float) -> str:
    if p < 0.10:
        return "low"
    if p < 0.30:
        return "medium"
    return "high"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


@app.on_event("startup")
def _load_artifacts():
    global MODEL, MODEL_INFO, TOP_FACTORS, THRESHOLD_REFERENCE, DECISION_POLICY

    if not MODEL_PATH.exists():
        MODEL = None
        return

    MODEL = joblib.load(MODEL_PATH)
    MODEL_INFO = _load_json(MODELINFO_PATH, {})
    TOP_FACTORS = _load_json(FI_PATH, {}).get("top_factors", [])
    THRESHOLD_REFERENCE = _load_json(THRESH_PATH, {}).get("threshold")
    DECISION_POLICY = _load_json(POLICY_PATH, DECISION_POLICY)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_INFO.get("model_version", "unknown"),
        "decision_policy": DECISION_POLICY,
        "threshold_reference": THRESHOLD_REFERENCE,
    }


@app.get("/model-info")
def model_info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MODEL_INFO


@app.get("/policy")
def policy():
    """
    Transparency endpoint so UIs/clients can display the operational decision policy.
    """
    return {
        "decision_policy": DECISION_POLICY,
        "threshold_reference": THRESHOLD_REFERENCE,
        "notes": (
            "Binary 'flagged' output is only defined for batch screening via the top-k policy. "
            "Single-record /predict returns a risk score only and must not be interpreted as a diagnosis."
        ),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Single-record risk scoring endpoint.
    Responsible behavior: returns risk_score + band; does NOT return a diagnosis label.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([payload.dict()])
    features = MODEL_INFO.get("features")
    if features:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing required fields: {missing}")
        df = df[features]

    risk = float(MODEL.predict_proba(df)[:, 1][0])

    return {
        "risk_score": risk,
        "risk_band": _risk_band(risk),
        "top_factors": TOP_FACTORS[:5],
        "notes": "Screening demo only. This is not medical advice or a clinical diagnosis.",
    }


@app.post("/screen", response_model=ScreenResponse)
def screen(payload: ScreenRequest):
    """
    Batch screening endpoint: flags exactly top-k fraction highest risk.
    This matches a capacity-constrained triage workflow.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not payload.records:
        raise HTTPException(status_code=422, detail="records must be a non-empty list")

    policy_obj = dict(DECISION_POLICY)
    top_k = payload.top_k if payload.top_k is not None else float(policy_obj.get("top_k", 0.10))
    if not (0.0 < top_k < 1.0):
        raise HTTPException(status_code=422, detail="top_k must be between 0 and 1")

    df = pd.DataFrame([r.dict() for r in payload.records])
    features = MODEL_INFO.get("features")
    if features:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing required fields: {missing}")
        df = df[features]

    risks = MODEL.predict_proba(df)[:, 1].astype(float)
    n = len(risks)
    k = int(np.ceil(top_k * n))

    # Flag exactly top-k by rank, deterministically break ties by original index (ascending).
    # lexsort sorts by last key first, so keys are (index asc, risk desc)
    order = np.lexsort((np.arange(n), -risks))
    flagged_idx = set(order[:k].tolist())

    results: list[dict] = []
    for i, r in enumerate(risks):
        results.append(
            {
                "index": i,
                "risk_score": float(r),
                "risk_band": _risk_band(float(r)),
                "flagged": i in flagged_idx,
            }
        )

    return {
        "policy": {**policy_obj, "top_k": float(top_k), "method": "exact_rank_tie_stable"},
        "flagged_count": k,
        "total": n,
        "results": results,
    }