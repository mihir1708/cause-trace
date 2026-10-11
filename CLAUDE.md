# CauseTrace — rules for Claude Code

CauseTrace predicts NTSB causal findings for an aviation incident from its facts, using
retrieval over earlier investigations. Leakage control is the core requirement. Read
`docs/DATA_LEAKAGE.md` and `docs/EVAL_PROTOCOL.md` before changing anything.

## Hard rules

1. **No target-event gold in model input.** The target event's probable cause
   (`narratives.narr_cause`), analysis narrative (`narratives.narr_accf`) and findings
   (the `findings` table and anything derived from it, including label codes) never enter
   model input. Model input for a target event is built only by
   `causetrace.packets.get_incident_packet`, which returns SAFE fields plus the sanitized
   factual narrative. `packets.py` never imports `causetrace.gold` and never selects
   GOLD_ONLY columns.
2. **Never fabricate or hand-edit results.** Metrics, reports and anything in `artifacts/`
   or `evals/` are produced by code from data. If a number is wrong, fix the code and
   regenerate it; never edit the output by hand or type numbers into docs or PRs that a
   script did not produce.
3. **All LLM calls go through the provider interface.** No module calls a model SDK or HTTP
   endpoint directly; every call goes through the single provider abstraction (to be added
   in `causetrace`), so it can be faked, budgeted and logged.
4. **Tests never use the network or paid APIs.** No real LLM, no paid API, no internet.
   Use fakes for providers. The only allowed connection is the local test PostgreSQL
   (`TEST_DATABASE_URL`).
5. **PostgreSQL is the only datastore.** No SQLite, no files-as-database, no other vector
   store. Vectors, when they arrive, live in PostgreSQL via pgvector. Schema changes are
   plain SQL migrations in `src/causetrace/migrations/`, applied by `causetrace.migrate`.
6. **The dataset split is by date and deterministic.** Held-out = `ev_date >= 2024-01-01`,
   history = earlier (`causetrace.split`). No random splits, no tuning on held-out events.
7. **Run `make check` before finishing** (ruff, ruff format check, mypy, pytest) and make it
   pass.

## Also

- Evaluation definitions in `docs/EVAL_PROTOCOL.md` may not change after results exist
  without a dated note in its change log.
- `data/raw/` and `.env` are gitignored. Never commit the NTSB download or secrets.
- `data/fixtures/` is a byte-for-byte sample of the `mdb-export` output; do not edit it by
  hand (regenerate with `scripts/make_fixture.py`).

## Layout

- `src/causetrace/` — package (`config`, `db`, `migrate`, `ingest`, `models`, `leakage`,
  `packets`, `gold`, `split`, `passages`, `embeddings`, `rerank`, `retrieval`, `index`;
  `llm/` provider interface (Anthropic, Ollama fallback, OpenAI, fake); `agent/` the
  LangGraph multi-hop agent; `evaluation/` metrics and runners; prompts in `prompts/*.md`;
  SQL in `migrations/`).
- Retrieval models (sentence-transformers, torch) are in the optional `ml` dependency
  group: `uv sync --group ml`. Tests use fakes (`tests/fakes.py`); the one `slow` test
  loads real models from the local Hugging Face cache only and skips when absent.
- `tests/` — pytest; needs PostgreSQL at `TEST_DATABASE_URL`
  (default `postgresql://causetrace:causetrace@localhost:5432/causetrace_test`).
- `scripts/` — `make_fixture.py`, `session_start.sh` (cloud SessionStart hook),
  `download_models.py`, `draft_questions.py` (paid LLM calls), `review_questions.py`,
  `eval_retrieval.py`, `eval_causal.py` (paid LLM calls), `grade_citations.py`.
- `evals/` — drafted and reviewed known-item questions; `artifacts/` — generated results
  and `ci_baseline.json` (the CI retrieval gate; change it only with `make update-baseline`).
- `docs/`, `data/fixtures/`, `data/raw/` (gitignored), `artifacts/`, `evals/`, `loadtests/`.

## Commands

```sh
make up                       # PostgreSQL 16 + pgvector via docker compose (local)
make migrate                  # apply SQL migrations to DATABASE_URL
make ingest DATA=data/fixtures  # load mdb-export CSVs, write artifacts/ingest_report.json
make models                   # download retrieval models into the HF cache (network)
make index                    # passages, lexical stats, embeddings (resumable)
make eval-retrieval           # artifacts/retrieval_results.{json,md}
make eval-causal              # six systems; artifacts/causal_results.{json,md} (paid, resumable)
make retrieval-gate           # CI gate: rrf vs artifacts/ci_baseline.json
make update-baseline          # deliberate baseline change (run on the fixture)
uv run --group ml python -m causetrace.agent <ev_id>   # agent result JSON with trace
make check                    # ruff, mypy, pytest
make down
```

In Claude Code cloud sessions there is no Docker; `scripts/session_start.sh` installs and
starts PostgreSQL 16 with pgvector via apt instead.
