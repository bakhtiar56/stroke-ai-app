# stroke_ai — Spec-driven Stroke Prediction (FastAPI + Streamlit + LightGBM)

Decision-support demo that predicts stroke probability from structured patient features.

## Disclaimer
This project is **NOT a medical device** and **NOT** intended for real clinical diagnosis or treatment decisions.

## Quickstart (Local)
1. Put your dataset CSV here (not committed):
   - `data/raw/stroke.csv`

2. Setup + train:
```bash
make setup
make train
```

3. Run services:
```bash
docker compose up --build
```

- API: http://localhost:8000 (docs at `/docs`)
- UI: http://localhost:8501

## Repo structure
- `docs/` specs (PRD, data contract, API spec, model spec, test plan)
- `src/` training + evaluation code
- `apps/api` FastAPI inference service
- `apps/ui` Streamlit UI
- `artifacts/` model + metrics (gitignored)