DATA ?= data/raw

-include .env
export

.PHONY: up down migrate ingest index models eval-retrieval retrieval-gate update-baseline check

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run python -m causetrace.migrate

ingest: migrate
	uv run python -m causetrace.ingest --data $(DATA) --report artifacts/ingest_report.json

index: migrate
	uv run --group ml python -m causetrace.index

models:
	uv run --group ml python scripts/download_models.py

eval-retrieval:
	uv run --group ml python scripts/eval_retrieval.py

# CI regression gate: rrf on the held-out events of the database (CI: the fixture).
retrieval-gate:
	uv run --group ml python scripts/eval_retrieval.py --gate artifacts/ci_baseline.json

# Deliberate baseline change: run on the fixture (make ingest DATA=data/fixtures && make index).
update-baseline:
	uv run --group ml python scripts/eval_retrieval.py --write-baseline artifacts/ci_baseline.json

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest
