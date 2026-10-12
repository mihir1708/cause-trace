"""Write the NTSB official findings for the README example to artifacts/readme_example.json.

The example is picked by a fixed rule (``causetrace.evaluation.readme.pick_example``): the
first candidate, in ev_id order, whose official cause findings (cm_inPc = 'T') are not only
"not determined" (05-00-00), since such a case leaves nothing to compare the answer with.
Skipped cases are recorded, so the README renders the same pick without a database.
GOLD_ONLY data, used for display only; nothing here feeds a model. Needs the full database
(DATABASE_URL).

Usage: uv run python scripts/export_readme_example.py
"""

import json
import sys

from causetrace.db import connect
from causetrace.evaluation.readme import ARTIFACTS, EXAMPLE_OFFICIAL, example_candidates

UNDETERMINED = "05-00-00"  # Not determined-Not determined-(general)

_OFFICIAL_SQL = """
SELECT label_code, min(finding_description)
FROM findings
WHERE ev_id = %s AND cm_inpc IS TRUE
GROUP BY label_code
ORDER BY label_code
"""


def main() -> int:
    causal_path = ARTIFACTS / "causal_results.json"
    causal = json.loads(causal_path.read_text(encoding="utf-8")) if causal_path.is_file() else None
    skipped: list[str] = []
    pick, rows = None, []
    with connect() as conn:
        for candidate in example_candidates(causal, ARTIFACTS.parent):
            rows = conn.execute(_OFFICIAL_SQL, (candidate["ev_id"],)).fetchall()
            if {label for label, _ in rows} <= {UNDETERMINED}:
                skipped.append(candidate["ev_id"])
                continue
            pick = candidate
            break
    if pick is None:
        print("no example case: run the causal evaluation first")
        return 1
    out = {
        "ev_id": pick["ev_id"],
        "skipped_undetermined": skipped,
        "official": [{"label": label, "description": desc} for label, desc in rows],
        "source": "findings with cm_inPc = 'T' (scripts/export_readme_example.py)",
    }
    path = ARTIFACTS / EXAMPLE_OFFICIAL
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {path}: {len(rows)} official findings for {pick['ev_id']} "
        f"(skipped {len(skipped)} not-determined cases)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
