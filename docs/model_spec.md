# Model Spec — Spec v1 (Structured Stroke Prediction)

## Objective
Binary classification: predict `stroke` ∈ {0,1}.

## Baseline Model
- `LogisticRegression` with:
  - class_weight="balanced"
  - preprocessing: impute + one-hot encode (sklearn ColumnTransformer)

## Primary Model (MVP)
- LightGBM classifier (via `lightgbm.LGBMClassifier`)
- Same preprocessing pipeline (impute + one-hot)

## Preprocessing (Must be identical in train/infer)
- Numeric columns:
  - Imputer: median
  - (Optional) StandardScaler for baseline only
- Categorical columns:
  - Imputer: most_frequent
  - OneHotEncoder(handle_unknown="ignore")

All steps stored in a single `sklearn.Pipeline` and saved as one artifact.

## Class Imbalance
- Use class weights or LightGBM parameters (e.g., `is_unbalance` or `scale_pos_weight`)
- Report positive class prevalence in report

## Evaluation Metrics (Report all)
- ROC-AUC
- PR-AUC
- Recall, precision, F1 at chosen threshold
- Confusion matrix at chosen threshold
- Calibration:
  - Brier score
  - reliability curve plot (saved)

## Threshold Selection
Choose threshold on validation set using rule:
- Find the smallest threshold achieving `recall >= 0.85`, and among those maximize precision.
If no threshold reaches recall 0.85, choose threshold maximizing F1.

Save selected threshold to `artifacts/threshold.json`.

## Explainability (Lightweight)
- Global permutation importance computed on validation set:
  - `sklearn.inspection.permutation_importance`
- Save top-N features to `artifacts/feature_importance.json`
- API returns the top factors from this saved file (not per-request recomputation).

## Artifacts (Required Outputs)
- `artifacts/model.joblib` (pipeline)
- `artifacts/threshold.json`
- `artifacts/metrics.json`
- `artifacts/feature_importance.json`
- `artifacts/plots/roc_curve.png`
- `artifacts/plots/pr_curve.png`
- `artifacts/plots/calibration.png`
- `MODEL_CARD.md` updated with latest metrics + limitations