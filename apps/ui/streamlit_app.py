from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Stroke AI (Screening Demo)", layout="wide")

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("Stroke AI — Screening Demo")
st.caption("Portfolio project. Screening demo only — not medical advice and not a clinical diagnosis.")


def _clean_records_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the dataframe JSON-safe for API calls:
    - coerce bmi to numeric (e.g. "N/A" -> NaN)
    - replace NaN/inf with None so JSON serialization is valid
    """
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)
    return df


with st.sidebar:
    st.subheader("API")
    st.write(f"Base URL: `{API_BASE}`")

    if st.button("Check /healthz"):
        try:
            r = requests.get(f"{API_BASE}/healthz", timeout=10)
            r.raise_for_status()
            st.json(r.json())
        except Exception as e:
            st.error(str(e))


tab1, tab2 = st.tabs(["Single prediction (/predict)", "Batch screening (/screen)"])


# -------------------------
# Single prediction
# -------------------------
with tab1:
    st.subheader("Single risk score")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("gender", ["Male", "Female", "Other"])
        age = st.number_input("age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
        hypertension = st.selectbox("hypertension", [0, 1])
        heart_disease = st.selectbox("heart_disease", [0, 1])

    with c2:
        ever_married = st.selectbox("ever_married", ["Yes", "No"])
        work_type = st.selectbox(
            "work_type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
        Residence_type = st.selectbox("Residence_type", ["Urban", "Rural"])
        smoking_status = st.selectbox(
            "smoking_status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        )

    with c3:
        avg_glucose_level = st.number_input(
            "avg_glucose_level", min_value=0.0, max_value=400.0, value=100.0, step=0.1
        )
        bmi_value = st.number_input("bmi", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        bmi_missing = st.checkbox("bmi is missing", value=False)

    payload = {
        "gender": gender,
        "age": float(age),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": float(avg_glucose_level),
        "bmi": None if bmi_missing else float(bmi_value),
        "smoking_status": smoking_status,
    }

    if st.button("Predict risk"):
        st.session_state["last_predict_payload"] = payload

    with st.expander("Payload being sent"):
        st.json(payload)

    if "last_predict_payload" in st.session_state:
        try:
            r = requests.post(
                f"{API_BASE}/predict", json=st.session_state["last_predict_payload"], timeout=20
            )
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
            else:
                data: dict[str, Any] = r.json()
                risk = float(data["risk_score"])
                threshold = float(data.get("screening_threshold_reference", 0.0))

                st.metric("risk_score", f"{risk:.8f}")
                st.metric("risk_band", data["risk_band"])

                if threshold > 0:
                    st.caption(f"Screening reference threshold: {threshold:.6f}")
                    st.write("Above screening reference threshold:", risk >= threshold)

                st.info(data.get("notes", ""))

                tf = data.get("top_factors") or []
                if tf:
                    st.write("Global feature importance (model-level, not patient-specific):")
                    st.dataframe(pd.DataFrame(tf))

                with st.expander("Raw response"):
                    st.json(data)
        except Exception as e:
            st.error(str(e))


# -------------------------
# Batch screening
# -------------------------
with tab2:
    st.subheader("Batch screening (flags top K by rank)")
    st.caption("Upload a CSV with the model feature columns. The app will flag exactly the top K highest-risk rows.")

    top_k = st.slider("top_k (fraction to flag)", min_value=0.01, max_value=0.50, value=0.10, step=0.01)

    st.markdown(
        """
**CSV columns expected:**
`gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status`

Extra columns (e.g. `id`) are OK — the API ignores them.
"""
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            df = _clean_records_df(df_raw)
            df = df.replace([np.nan, np.inf, -np.inf], None)


            st.write("Preview:")
            st.dataframe(df.head(20))

            if st.button("Run screening"):
                records = df.to_dict(orient="records")
                body = {"top_k": float(top_k), "records": records}

                r = requests.post(f"{API_BASE}/screen", json=body, timeout=60)
                if r.status_code != 200:
                    st.error(f"API error {r.status_code}: {r.text}")
                else:
                    data = r.json()
                    st.write("Policy:")
                    st.json(data.get("policy", {}))

                    results = pd.DataFrame(data["results"])
                    results = results.sort_values(["flagged", "risk_score"], ascending=[False, False])

                    st.success(
                        f"Flagged {data['flagged_count']} of {data['total']} rows "
                        f"({data['flagged_count']/max(1,data['total']):.1%})."
                    )
                    st.dataframe(results, use_container_width=True)

                    # Merge back to original rows for download (preserve original ordering)
                    out = df_raw.copy()
                    res_by_index = pd.DataFrame(data["results"]).sort_values("index")
                    out["risk_score"] = res_by_index["risk_score"].to_numpy()
                    out["risk_band"] = res_by_index["risk_band"].to_numpy()
                    out["flagged"] = res_by_index["flagged"].to_numpy()

                    buf = io.StringIO()
                    out.to_csv(buf, index=False)
                    st.download_button(
                        "Download results CSV",
                        data=buf.getvalue(),
                        file_name="stroke_screening_results.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(str(e))