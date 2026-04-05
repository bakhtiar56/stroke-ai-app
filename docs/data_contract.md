# Data Contract (Spec v1)

## Data Source
Public structured "stroke prediction" CSV dataset.

For this project, training expects the file:
- `data/raw/stroke.csv`

## Required Columns
Target:
- `stroke` (int) in {0,1}

Features (expected):
- `gender` (str) in {"Male","Female","Other"}  *(dataset-dependent; if absent, update allowed set)*
- `age` (float) range [0, 120]
- `hypertension` (int) in {0,1}
- `heart_disease` (int) in {0,1}
- `ever_married` (str) in {"Yes","No"}
- `work_type` (str) in {"Private","Self-employed","Govt_job","children","Never_worked"}
- `Residence_type` (str) in {"Urban","Rural"}
- `avg_glucose_level` (float) range [0, 500]  *(wide range; validate non-negative)*
- `bmi` (float) range [0, 100] and nullable allowed
- `smoking_status` (str) in {"formerly smoked","never smoked","smokes","Unknown"}

Optional (ignored if present):
- `id` (int)

## Missingness Rules
- `bmi` may be missing; it will be imputed (median).
- Other required fields: missing values are allowed but will be imputed; missingness rate is reported in training report.

## Validation Rules
Hard failures:
- Missing required columns (except optional `id`)
- Target not in {0,1}
- Negative values for `age`, `avg_glucose_level`, or `bmi`

Soft checks (warnings in report):
- Outliers near upper bounds (e.g., `avg_glucose_level` > 300)
- High missingness in any feature (> 20%)

## Train/Val/Test Split
- Stratified by `stroke`
- Fixed random seed stored in config