# Stroke AI (Structured Data) — PRD (Spec v1)

## Purpose
Build an end-to-end ML application (local + Docker) that predicts the probability of stroke from structured patient features and presents results in:
- FastAPI (inference API)
- Streamlit (simple UI)

This is a portfolio project demonstrating production-oriented ML engineering: data validation, reproducible training, evaluation, calibration, testing, and API design.

## Intended Users
- Primary: Hiring managers / reviewers evaluating entry-level ML/AI engineering skills
- Secondary: Developers experimenting with model training/inference workflows

## Explicit Safety Disclaimer
This software is **NOT** a medical device and **NOT** for real clinical diagnosis or treatment decisions.
It is a learning project / demo decision-support tool built on public datasets.

## In Scope (MVP)
1. Train a model from a public structured dataset (CSV).
2. Validate input data with a strict schema (fail fast).
3. Produce evaluation reports (AUC, PR-AUC, calibration, confusion matrix).
4. Save a single versioned model artifact usable for inference.
5. Serve predictions via FastAPI:
   - `/predict`, `/model-info`, `/healthz`
6. Provide a Streamlit UI for interactive prediction.
7. Dockerized local run.

## Out of Scope (v1)
- Imaging (CT/MRI) stroke diagnosis
- Real clinical deployment, HIPAA, EHR integration
- Real-time streaming inference
- Federated learning or on-device ML

## Success Criteria
- One-command training produces artifacts in `artifacts/` and a metrics report.
- API and UI run locally (and in Docker) and make consistent predictions.
- Tests pass in CI: lint + unit + API contract + training smoke test.
- Documentation explains limitations, intended use, and performance.

## Non-Functional Requirements
- Reproducibility: deterministic seeds; model + config saved together.
- Maintainability: modular code with type hints and tests.
- Observability: API logs inputs shape + model version (no PHI assumption; still avoid logging raw payloads).
- Portability: runs on CPU without GPUs.

## Milestones
- M0: Specs committed (this doc + data contract + API spec + model spec + test plan)
- M1: Data loader + schema validation
- M2: Baseline model + evaluation report
- M3: LightGBM model + calibration + threshold selection
- M4: FastAPI inference service
- M5: Streamlit UI
- M6: Docker + CI