"""Evaluate retrieval on reviewed questions and held-out incidents; write
artifacts/retrieval_results.{json,md}. With --gate, run only the CI regression check.

Usage:
  uv run --group ml python scripts/eval_retrieval.py [--limit N] [--seed S]
  uv run --group ml python scripts/eval_retrieval.py --gate artifacts/ci_baseline.json
  uv run --group ml python scripts/eval_retrieval.py --write-baseline artifacts/ci_baseline.json
"""

import sys

from causetrace.evaluation.retrieval_eval import main

if __name__ == "__main__":
    sys.exit(main())
