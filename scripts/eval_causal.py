"""Compare the six causal systems on held-out incidents (docs/EVAL_PROTOCOL.md).

LLM systems call Claude (paid; ANTHROPIC_API_KEY in .env) and stop at MAX_USD; results are
cached in artifacts/runs/ so an interrupted run resumes. Writes artifacts/causal_results.*
and samples evals/citation_audit.csv once rag_verify has answered every case.

Usage:
  uv run --group ml python scripts/eval_causal.py --cases 100 --workers 4
  uv run --group ml python scripts/eval_causal.py --systems rag rag_verify --model claude-sonnet-5-5
  uv run --group ml python scripts/eval_causal.py --report-only
"""

import sys

from causetrace.evaluation.causal_eval import main

if __name__ == "__main__":
    sys.exit(main())
