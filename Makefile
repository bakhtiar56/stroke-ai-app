.PHONY: setup test lint train run-api run-ui docker-build docker-up

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check .

train:
	. .venv/bin/activate && python -m stroke_ai.models.train

run-api:
	. .venv/bin/activate && uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	. .venv/bin/activate && streamlit run apps/ui/app.py

docker-build:
	docker compose build

docker-up:
	docker compose up --build