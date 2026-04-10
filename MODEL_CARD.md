# Model Card — Stroke AI (Screening Demo)

## 1) Summary
**Model name:** Stroke AI (LightGBM)  
**Version:** 0.1.0  
**Task:** Predict a **stroke risk score** from structured patient features.  
**Output type:** Probability-like score in **[0, 1]** used for **ranking** (screening), not diagnosis.

This project is built as a **capacity-based screening demo**: in a batch of records, the system flags the **top K% highest-risk** rows (default `top_k = 0.10`) to simulate limited follow-up capacity.

---

## 2) Intended Use
### Intended use
- Educational / portfolio demonstration of:
  - Spec-driven ML development
  - Artifact-based training → serving workflow
  - Capacity-constrained decision policy (top-K ranking)
  - FastAPI inference + Streamlit UI

### Not intended use
- **Not a medical device**
- **Not** for clinical diagnosis, treatment decisions, or real patient care
- Not validated on local hospital populations, not calibrated for real-world prevalence

---

## 3) Data
### Dataset
- The model is trained using a structured tabular dataset with a binary target (`stroke`).
- Expected input columns:
  - `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`,
    `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`

### Data limitations
- The dataset may not represent real clinical populations or local demographics.
- Some variables (e.g., `bmi`, smoking) may contain missing values or coarse categories (`Unknown`).
- Potential for **label noise** and **unobserved confounders**.

---

## 4) Model Details
### Model type
- **LightGBM classifier** inside a scikit-learn Pipeline

### Preprocessing
- Numeric features: median imputation
- Categorical features: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`)

### Class imbalance handling
- **RandomOverSampler applied to training split only** (to reduce imbalance effects without leaking test information).

---

## 5) Decision Policy (Operational Definition)
This system is designed for **screening under limited capacity**.

### Batch screening policy (`/screen`)
- Policy: `top_k`
- Method: `exact_rank` (stable ranking / deterministic tie behavior)
- Default: `top_k = 0.10` (flag the top 10% highest-risk rows)

### Reference threshold
To make decisions explainable, training stores a **screening reference threshold**:
- `threshold_reference`: score at the K-th ranked position on the test set for `top_k = 0.10`

This is a **ranking threshold**, not a clinical diagnostic cutoff.

---

## 6) Performance (Test Set)
Metrics computed on a held-out test set. These values are taken from training artifacts (see `/model-info`).

- ROC-AUC: **0.7967**
- PR-AUC: **0.1999**
- Brier score: **0.0514**

At `top_k = 0.10` flagged (capacity-based policy):
- Precision: **0.2078**
- Recall: **0.4211**
- F1: **0.2783**
- Flag rate (test): **0.1004**

Interpretation (capacity-based):
- When flagging the top 10% highest-risk records, about **1 in 5** flagged records are true positives on this dataset.
- About **42%** of all true stroke cases are captured within the top 10% flagged set.

---

## 7) Explainability / Feature Importance
The UI may display **global feature importance** (model-level), which is **not patient-specific**.

In this baseline, **age** is often the dominant signal, which can be influenced by:
- true correlation in the dataset,
- one-hot encoding spreading importance across many categorical dummy variables,
- or dataset-specific biases.

For better portfolio-quality importance, permutation importance on the full pipeline is recommended.

---

## 8) Fairness, Bias, and Ethical Considerations
- Features like `age`, `gender`, and proxy variables correlated with socioeconomic factors may lead to disparate impacts.
- The dataset may underrepresent some groups; performance may vary across subpopulations.
- The model does not incorporate clinical context (labs, imaging, history), so it should not be used for real patient decisions.

---

## 9) Safety / Monitoring (If this were ever deployed)
If used beyond a demo, you would monitor:
- Flag rate over time (should remain near configured `top_k`)
- Score distribution drift
- Feature missingness drift (e.g., BMI missing rate changes)
- Calibration drift (Brier score, reliability diagrams)
- Subgroup performance (age buckets, gender categories) with careful governance

---

## 10) How to Reproduce
1. Place dataset at: `data/raw/stroke.csv`
2. Train and write artifacts:
   ```bash
   python -m stroke_ai.models.train
   ```
3. Run API:
   ```bash
   uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
   ```
4. Run UI:
   ```bash
   set API_BASE_URL=http://127.0.0.1:8000
   streamlit run apps/ui/streamlit_app.py
   ```

---

## 11) Contact
Portfolio project by repository owner. Use GitHub Issues for questions or improvements.