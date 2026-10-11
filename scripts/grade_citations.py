"""Grade sampled rag_verify citations in the terminal: 2 supports, 1 partly, 0 does not.

Shows each ungraded row of evals/citation_audit.csv with the incident packet and the cited
passage, and saves the grade immediately. Quit any time with q; rerun to resume.

Usage: uv run python scripts/grade_citations.py [--audit PATH]
"""

import argparse
import shutil
from pathlib import Path

from causetrace.agent.graph import describe_packet
from causetrace.db import connect
from causetrace.evaluation.citation_audit import DEFAULT_AUDIT, grade_loop, read_audit
from causetrace.packets import get_incident_packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    if not read_audit(args.audit):
        raise SystemExit(f"no citations in {args.audit}; run scripts/eval_causal.py first")
    width = min(shutil.get_terminal_size((100, 20)).columns, 120)
    with connect(args.database_url) as conn:

        def incident(ev_id: str) -> str:
            return describe_packet(get_incident_packet(ev_id, conn))

        def passage(passage_id: str) -> str | None:
            row = conn.execute(
                "SELECT content FROM passages WHERE passage_id = %s", (passage_id,)
            ).fetchone()
            return row[0] if row else None

        recorded = grade_loop(args.audit, incident, passage, width=width)
    rows = read_audit(args.audit)
    graded = sum(r.get("grade") in {"0", "1", "2"} for r in rows)
    print(f"recorded {recorded} now; {graded} of {len(rows)} graded in {args.audit}")


if __name__ == "__main__":
    main()
