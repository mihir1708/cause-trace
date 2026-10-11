"""Review drafted questions in the terminal: keep, edit or drop each one.

Shows each draft from evals/questions_draft.jsonl beside its source passage and appends
the decision to evals/questions_reviewed.jsonl. Quit any time with q; rerun to resume.

Usage: uv run python scripts/review_questions.py [--drafts PATH] [--out PATH]
"""

import argparse
import shutil
from pathlib import Path

from causetrace.db import connect
from causetrace.evaluation.questions import read_jsonl, review_loop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drafts", type=Path, default=Path("evals/questions_draft.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("evals/questions_reviewed.jsonl"))
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    drafts = read_jsonl(args.drafts)
    if not drafts:
        raise SystemExit(f"no drafts in {args.drafts}; run scripts/draft_questions.py first")
    width = min(shutil.get_terminal_size((100, 20)).columns, 120)
    with connect(args.database_url) as conn:

        def passage_text(passage_id: str) -> str | None:
            row = conn.execute(
                "SELECT content FROM passages WHERE passage_id = %s", (passage_id,)
            ).fetchone()
            return row[0] if row else None

        recorded = review_loop(drafts, args.out, passage_text, width=width)
    total = len(read_jsonl(args.out))
    print(f"recorded {recorded} now; {total} of {len(drafts)} reviewed in {args.out}")


if __name__ == "__main__":
    main()
