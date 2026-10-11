DATA ?= data/raw

-include .env
export

.PHONY: up down migrate ingest check

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run python -m causetrace.migrate

ingest: migrate
	uv run python -m causetrace.ingest --data $(DATA) --report artifacts/ingest_report.json

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest
