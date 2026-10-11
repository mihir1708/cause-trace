DATA ?= data/raw

-include .env
export

.PHONY: up down migrate ingest index models eval-retrieval eval-causal eval-causal-dev retrieval-gate update-baseline serve api loadtest readme check

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

eval-causal:
	uv run --group ml python scripts/eval_causal.py

eval-causal-dev:
	uv run --group ml python scripts/eval_causal.py --split dev

# HTTP API on the host (needs the ml group for the retrieval models).
serve:
	uv run --group ml causetrace serve

# HTTP API in Docker (docker-compose.yml, profile "api"); add --profile tracing for Jaeger.
api:
	docker compose --profile api up -d --build --wait api

# Load test /v1/search against a running API: 1, 10, 25, 50 users, 60 s each.
# Writes artifacts/load_results.{json,md}.
LOAD_HOST ?= http://localhost:8000
loadtest:
	uv run --group load python loadtests/run.py --host $(LOAD_HOST)

# Results tables in README.md, regenerated from artifacts/.
readme:
	uv run python scripts/render_readme.py

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
