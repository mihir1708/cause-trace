"""Write the README example's case and the NTSB's official findings to
artifacts/readme_example.json.

The example is chosen to show a case the system got right, by a fixed rule: the first
held-out candidate (``causetrace.evaluation.readme.example_candidates``), in ev_id order,
whose suggested causes exactly match its official cause findings (cm_inPc = 'T'), with three
or more. The file also records how many candidates there were and how many had every
suggestion confirmed, so the README states plainly that the example is not typical.
GOLD_ONLY data, used for display only; nothing here feeds a model. Needs the full database
(DATABASE_URL).

Usage: uv run python scripts/export_readme_example.py
"""

import json
import sys

from causetrace.db import connect
from causetrace.evaluation.readme import ARTIFACTS, EXAMPLE_OFFICIAL, example_candidates

MIN_EXACT = 3

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
    candidates = example_candidates(causal, ARTIFACTS.parent)
    pick, pick_rows, all_confirmed, n_exact = None, [], 0, 0
    with connect() as conn:
        for candidate in candidates:
            rows = conn.execute(_OFFICIAL_SQL, (candidate["ev_id"],)).fetchall()
            gold = {label for label, _ in rows}
            suggested = {f["label"] for f in candidate["findings"]}
            if suggested <= gold:
                all_confirmed += 1
            if suggested == gold and len(suggested) >= MIN_EXACT:
                n_exact += 1
                if pick is None:
                    pick, pick_rows = candidate, rows
    if pick is None:
        print("no example case: run the causal evaluation first")
        return 1
    out = {
        "ev_id": pick["ev_id"],
        "official": [{"label": label, "description": desc} for label, desc in pick_rows],
        "n_candidates": len(candidates),
        "n_all_confirmed": all_confirmed,
        "n_exact": n_exact,
        "source": "findings with cm_inPc = 'T' (scripts/export_readme_example.py)",
    }
    path = ARTIFACTS / EXAMPLE_OFFICIAL
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {path}: example {pick['ev_id']}; {all_confirmed} of {len(candidates)} "
        f"candidates had every suggestion confirmed, {n_exact} exact matches of {MIN_EXACT}+"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
