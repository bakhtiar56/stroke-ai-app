# Test Plan — Spec v1

## Unit Tests
1. Data schema validation
   - Missing required columns => fail
   - Negative values in numeric fields => fail
2. Feature pipeline
   - Pipeline can fit and predict on a 5-row synthetic dataset
3. Threshold selection
   - Deterministic output given fixed y_true, y_proba

## Integration Tests
1. Training smoke test
   - Run training on a small sample (or with a flag `--smoke`) and verify artifacts exist
2. API contract test
   - Start FastAPI app (TestClient)
   - `GET /healthz` returns model_loaded true after loading artifacts
   - `POST /predict` returns required fields with correct types/ranges

## CI Gates
- ruff (lint)
- pytest
- (optional) mypy for type checking