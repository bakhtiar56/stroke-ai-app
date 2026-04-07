.PHONY: setup test lint train run-api run-ui docker-build docker-up

ifeq ($(OS),Windows_NT)
  PY := .\.venv\Scripts\python.exe
  PIP := .\.venv\Scripts\pip.exe
else
  PY := ./.venv/bin/python
  PIP := ./.venv/bin/pip
endif

setup:
	python -m venv .venv
	$(PY) -m pip install -U pip
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

train:
	$(PY) -m stroke_ai.models.train

run-api:
	$(PY) -m uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	$(PY) -m streamlit run apps/ui/app.py

docker-build:
	docker compose build

docker-up:
	docker compose up --build