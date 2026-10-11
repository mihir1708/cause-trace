DATA ?= data/raw

-include .env
export

.PHONY: up down migrate ingest index models check

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

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest
