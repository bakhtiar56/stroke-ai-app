# API Spec (FastAPI) — Spec v1

Base URL (local):
- http://localhost:8000

## GET /healthz
Response 200 JSON:
- `status`: "ok"
- `model_loaded`: boolean
- `model_version`: string

## GET /model-info
Response 200 JSON:
- `model_version`: string
- `trained_at`: ISO timestamp
- `features`: list[str]
- `threshold`: float
- `metrics`: object (selected summary metrics)

## POST /predict
### Request JSON
A single patient record with the feature fields (no target).

Example:
```json
{
  "gender": "Male",
  "age": 67,
  "hypertension": 1,
  "heart_disease": 0,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

### Response JSON
- `probability`: float (0..1)
- `risk_band`: "low" | "medium" | "high"
- `threshold_used`: float
- `prediction`: int (0 or 1 based on threshold)
- `top_factors`: list of objects with:
  - `feature`: string
  - `importance`: float (permutation importance magnitude; normalized)

Example:
```json
{
  "probability": 0.37,
  "risk_band": "medium",
  "threshold_used": 0.22,
  "prediction": 1,
  "top_factors": [
    {"feature": "age", "importance": 0.31},
    {"feature": "avg_glucose_level", "importance": 0.22},
    {"feature": "hypertension", "importance": 0.10}
  ]
}
```

## Error Handling
- 422: validation errors (Pydantic)
- 500: model not loaded or internal inference error (includes request_id)