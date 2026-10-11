"""Human audit of rag_verify citations: sample, grade (2/1/0) and summarize.

``evals/citation_audit.csv`` holds 50 citations sampled from rag_verify answers across
cases; ``scripts/grade_citations.py`` fills its ``grade`` column one row at a time:
2 = the passage supports the label for the incident, 1 = partly, 0 = does not.
"""

import csv
import os
import random
import textwrap
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_AUDIT = Path("evals/citation_audit.csv")
AUDIT_SIZE = 50
COLUMNS = (
    "audit_id",
    "ev_id",
    "label",
    "short_name",
    "passage_id",
    "rationale",
    "grade",
    "graded_at",
)
GRADES = {"2": 2, "1": 1, "0": 0}


def read_audit(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_audit(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Rewrite the whole file atomically, so a crash never loses earlier grades."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
    os.replace(tmp, path)


def write_audit_sample(
    path: Path,
    answers: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, str],
    seed: int,
    n: int = AUDIT_SIZE,
) -> bool:
    """Sample ``n`` (finding, citation) pairs from rag_verify answers; never overwrite.

    Returns True when a new sample was written.
    """
    if path.is_file():
        return False
    pool = sorted(
        {
            (r["ev_id"], str(f.get("label", "")), pid, str(f.get("rationale") or ""))
            for r in answers
            for f in r.get("findings") or []
            for pid in f.get("citations") or []
        }
    )
    if not pool:
        return False
    picked = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
    write_audit(
        path,
        [
            {
                "audit_id": i,
                "ev_id": ev,
                "label": label,
                "short_name": taxonomy.get(label, ""),
                "passage_id": pid,
                "rationale": rationale,
                "grade": "",
                "graded_at": "",
            }
            for i, (ev, label, pid, rationale) in enumerate(picked, start=1)
        ],
    )
    return True


def audit_summary(path: Path, verifier_support_rate: float | None) -> dict[str, Any] | None:
    """Human citation precision (share graded 2, and 1 or 2) next to the verifier's rate."""
    rows = read_audit(path)
    if not rows:
        return None
    grades = [int(r["grade"]) for r in rows if r.get("grade") in GRADES]
    return {
        "path": str(path),
        "sampled": len(rows),
        "graded": len(grades),
        "precision_strict": sum(g == 2 for g in grades) / len(grades) if grades else None,
        "precision_lenient": sum(g >= 1 for g in grades) / len(grades) if grades else None,
        "verifier_support_rate": verifier_support_rate,
    }


def grade_loop(
    path: Path,
    incident_text: Callable[[str], str],
    passage_text: Callable[[str], str | None],
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    width: int = 100,
) -> int:
    """Show each ungraded citation and record 2/1/0; ``s`` skips, ``q``/EOF stops.

    The file is rewritten after every grade, so the next run resumes where this one stopped.
    Returns the number of grades recorded now.
    """
    rows = read_audit(path)
    recorded = 0
    for row in rows:
        if row.get("grade") in GRADES:
            continue
        done = sum(r.get("grade") in GRADES for r in rows)
        show("=" * width)
        show(f"[{done + 1}/{len(rows)}] audit {row['audit_id']}  incident {row['ev_id']}")
        show("-" * width)
        show("INCIDENT:")
        show(textwrap.shorten(incident_text(row["ev_id"]), 1500, placeholder=" ..."))
        show("-" * width)
        show(f"FINDING: {row['label']}  {row['short_name']}")
        show(f"RATIONALE: {row['rationale']}")
        show("-" * width)
        show(f"CITED PASSAGE [{row['passage_id']}]:")
        show(textwrap.fill(passage_text(row["passage_id"]) or "(passage not found)", width))
        while True:
            try:
                answer = ask("2 supports / 1 partly / 0 does not / s skip / q quit > ").strip()
            except EOFError:
                return recorded
            if answer.lower() == "q":
                return recorded
            if answer.lower() == "s" or answer in GRADES:
                break
        if answer in GRADES:
            row["grade"] = answer
            row["graded_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            write_audit(path, rows)
            recorded += 1
    return recorded
