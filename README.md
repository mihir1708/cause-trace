# cause-trace

CauseTrace predicts NTSB causal findings for an aviation incident from its facts, using
retrieval over earlier investigations. Leakage control is the core requirement: see
[`docs/DATA_LEAKAGE.md`](docs/DATA_LEAKAGE.md) and
[`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md). Contributor rules are in
[`CLAUDE.md`](CLAUDE.md).

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```sh
cp .env.example .env
uv sync
make up                          # PostgreSQL 16 + pgvector
make ingest DATA=data/fixtures   # 300-event fixture; report in artifacts/ingest_report.json
uv sync --group ml               # sentence-transformers + torch for retrieval
make models                      # download the embedding and reranking models
make index                       # passages + embeddings; resumes if interrupted
make check                       # ruff, mypy, pytest
```

For the full dataset, export the `events`, `aircraft`, `narratives` and `Findings` tables of
`avall.mdb` with `mdb-export` into `data/raw/` and run `make ingest`.

## Retrieval evaluation

```sh
uv run python scripts/draft_questions.py    # paid: needs ANTHROPIC_API_KEY in .env; capped by MAX_USD
uv run python scripts/review_questions.py   # keep / edit / drop each draft (resumable)
make eval-retrieval                         # writes artifacts/retrieval_results.{json,md}
```
