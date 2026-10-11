"""Draft known-item questions with the LLM (costs money; needs ANTHROPIC_API_KEY in .env).

Samples passages from pre-2024 events (stratified by section type and year, fixed seed),
asks the LLM for one investigator question per passage, and appends them to
evals/questions_draft.jsonl. Resumable; stops when MAX_USD (default 10) is reached.

Usage: uv run python scripts/draft_questions.py [--n 220] [--seed 42] [--out PATH]
"""

import argparse
from pathlib import Path

from causetrace.db import connect
from causetrace.evaluation.questions import DEFAULT_N, DEFAULT_SEED, draft_all, sample_passages
from causetrace.llm import BudgetExceededError
from causetrace.llm.anthropic_provider import AnthropicProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("evals/questions_draft.jsonl"))
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    with connect(args.database_url) as conn:
        passages = sample_passages(conn, args.n, args.seed)
    provider = AnthropicProvider()
    print(f"{len(passages)} passages sampled; drafting with {provider.model}")
    try:
        written, failed = draft_all(provider, passages, args.out)
    except BudgetExceededError as err:
        print(f"stopped: {err}")
        written, failed = -1, -1
    t = provider.tracker
    print(
        f"written {written}, failed {failed}; {len(t.calls)} calls, "
        f"{t.input_tokens} input / {t.output_tokens} output tokens, ${t.spent_usd:.4f}"
    )


if __name__ == "__main__":
    main()
