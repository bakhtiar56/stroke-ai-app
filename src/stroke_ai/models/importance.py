from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def save_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    out_path: Path,
    scoring: str = "roc_auc",
    n_repeats: int = 5,
    random_state: int = 42,
    top_n: int = 15,
) -> dict[str, Any]:
    """
    Computes permutation importance on the *full pipeline* (preprocess + model).
    This is typically more reliable than LightGBM split/gain importance when one-hot encoding is used.
    """
    r = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    imps = np.asarray(r.importances_mean, dtype=float)
    imps = np.maximum(imps, 0.0)

    denom = float(imps.sum()) if float(imps.sum()) > 0 else 1.0
    imps = imps / denom

    rows = [{"feature": str(f), "importance": float(i)} for f, i in zip(X.columns, imps)]
    rows.sort(key=lambda d: d["importance"], reverse=True)

    payload = {"method": "permutation_importance", "scoring": scoring, "top_factors": rows[:top_n]}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload