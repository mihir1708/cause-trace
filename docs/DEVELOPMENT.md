# Development guide

Setup, the API, the command-line tool, the evaluations, the project rules and the code
layout. The [README](../README.md) is the overview.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```sh
cp .env.example .env               # ANTHROPIC_API_KEY for the agent; MAX_USD caps spend
uv sync --group ml                 # app + sentence-transformers/torch for retrieval
make up                            # PostgreSQL 16 + pgvector
make ingest DATA=data/fixtures     # the committed fixture sample
make models                        # download the embedding and reranking models
make index                         # passages + embeddings; resumes if interrupted
make check                         # ruff, ruff format check, mypy, pytest
```

For the full dataset, download `avall.zip` from the NTSB (link and checksums in
`data_manifest.json`), export the `events`, `aircraft`, `narratives` and `Findings` tables
of `avall.mdb` with `mdb-export` into `data/raw/` and run `make ingest`.

In Claude Code cloud sessions there is no Docker; `scripts/session_start.sh` installs and
starts PostgreSQL 16 with pgvector via apt instead.

## API

```sh
make serve                         # uvicorn on http://127.0.0.1:8000 (or: make api, in Docker)
curl -s localhost:8000/health
curl -s localhost:8000/v1/search -H 'content-type: application/json' \
  -d '{"query": "engine lost power after takeoff", "before_date": "2024-01-01", "top_k": 5}'
curl -s localhost:8000/v1/analyze -H 'content-type: application/json' -d '{"ev_id": "<ev_id>"}'
curl -s localhost:8000/v1/runs/<run_id>
```

- `POST /v1/search`: hybrid search over history passages (`mode`: dense, lexical, rrf,
  rrf_rerank). If the reranker fails, the response uses rrf and says so in `fallback`.
- `POST /v1/analyze`: the agent on a stored event (`ev_id`) or a free-text `packet`.
  Packets are length-capped and checked for prompt injection (400). Findings without a
  valid citation are dropped; `grounding_score` is the share kept. An LLM timeout, overload
  or missing key returns 503; invalid model JSON is retried once, then the run abstains.
- `GET /v1/runs/{id}`: the stored run with its full node trace.
- `GET /health`: 503 when the database is down. A busy connection pool answers 503
  "server busy" with `Retry-After`.

Interactive docs at `/docs`. Traces go to the console by default; for Jaeger run
`docker compose --profile api --profile tracing up -d` with `CAUSETRACE_TRACING=otlp` and
open http://localhost:16686.

## Command line

```sh
uv run --group ml causetrace ask <ev_id>          # agent answer with citations and trace
uv run --group ml causetrace eval retrieval       # = scripts/eval_retrieval.py
uv run --group ml causetrace eval causal -- --report-only   # = scripts/eval_causal.py (paid)
uv run --group ml causetrace serve                # the API
```

## Evaluations

```sh
make eval-retrieval                # artifacts/retrieval_results.{json,md}
make eval-causal-dev               # tune LLM cut-offs on 2023 events (paid)
make eval-causal                   # six systems on held-out events (paid, resumable)
uv run python scripts/grade_citations.py          # grade the 50 sampled citations
make loadtest                      # /v1/search at 1, 10, 25, 50 users (API must be running)
make readme                        # regenerate the README results from artifacts/
make retrieval-gate                # CI gate: rrf vs artifacts/ci_baseline.json
make update-baseline               # deliberate baseline change (run on the fixture)
```

## Project rules

1. **No target-event gold in model input.** The target event's probable cause
   (`narratives.narr_cause`), analysis narrative (`narratives.narr_accf`) and findings
   (the `findings` table and anything derived from it, including label codes) never enter
   model input. Model input for a target event is built only by
   `causetrace.packets.get_incident_packet`, which returns SAFE fields plus the sanitized
   factual narrative. `packets.py` never imports `causetrace.gold` and never selects
   GOLD_ONLY columns. See [`DATA_LEAKAGE.md`](DATA_LEAKAGE.md).
2. **Never fabricate or hand-edit results.** Metrics, reports and anything in `artifacts/`
   or `evals/` are produced by code from data. If a number is wrong, fix the code and
   regenerate it; never edit the output by hand or type result numbers into docs or PRs.
   The README results are generated (`make readme`) and a test fails when they are stale.
3. **All LLM calls go through the provider interface** (`causetrace.llm`). No module calls
   a model SDK or HTTP endpoint directly, so every call can be faked, budgeted and logged.
4. **Tests never use the network or paid APIs.** Providers are faked; the only allowed
   connection is the local test PostgreSQL (`TEST_DATABASE_URL`).
5. **PostgreSQL is the only datastore.** Vectors live in PostgreSQL via pgvector. Schema
   changes are plain SQL migrations in `src/causetrace/migrations/`, applied by
   `causetrace.migrate`.
6. **The dataset split is by date and deterministic.** Held-out = `ev_date >= 2024-01-01`,
   history = earlier (`causetrace.split`). No random splits, no tuning on held-out events.
7. **`make check` passes before every change is merged** (ruff, ruff format check, mypy,
   pytest).

Also:

- Evaluation definitions in [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) may not change after
  results exist without a dated note in its change log.
- `data/raw/` and `.env` are gitignored. Never commit the NTSB download or secrets.
- `data/fixtures/` is a byte-for-byte sample of the `mdb-export` output; regenerate it
  with `scripts/make_fixture.py`, never edit it by hand.

## Layout

- `src/causetrace/`: `config`, `db`, `migrate`, `ingest`, `models`, `leakage`, `packets`,
  `gold`, `split`, `passages`, `embeddings`, `rerank`, `retrieval`, `index`, `guardrails`,
  `telemetry`, `cli`; `llm/` the provider interface (Anthropic, Ollama, OpenAI, fake, and
  `ReliableProvider` for serving); `agent/` the LangGraph agent; `api/` the FastAPI
  service; `evaluation/` metrics, runners and report renderers; `prompts/*.md`;
  `migrations/*.sql`.
- `tests/`: pytest; needs PostgreSQL at `TEST_DATABASE_URL` (default
  `postgresql://causetrace:causetrace@localhost:5432/causetrace_test`). Retrieval models are
  faked (`tests/fakes.py`); the one `slow` test loads real models from the local Hugging
  Face cache and skips when they are absent.
- `scripts/`: `make_fixture.py`, `session_start.sh`, `download_models.py`,
  `draft_questions.py` (paid), `review_questions.py`, `eval_retrieval.py`,
  `eval_causal.py` (paid), `grade_citations.py`, `render_readme.py`.
- `loadtests/`: `locustfile.py` and `run.py` for `make loadtest` (Locust in the `load`
  dependency group).
- `evals/`: known-item questions, frozen case lists and the citation audit.
- `artifacts/`: generated results, cached runs and `ci_baseline.json`.
- `docs/`: [`DATA_LEAKAGE.md`](DATA_LEAKAGE.md), [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md),
  [`DECISIONS.md`](DECISIONS.md) and this guide.
