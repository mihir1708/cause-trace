"""Write the NTSB official findings for the README example to artifacts/readme_example.json.

The example is picked by a fixed rule (``causetrace.evaluation.readme.pick_example``); this
reads its cause findings (cm_inPc = 'T') from the database so the README can show what the
NTSB concluded next to the suggested causes. GOLD_ONLY data, used for display only; nothing
here feeds a model. Needs the full database (DATABASE_URL).

Usage: uv run python scripts/export_readme_example.py
"""

import json
import sys

from causetrace.db import connect
from causetrace.evaluation.readme import ARTIFACTS, EXAMPLE_OFFICIAL, pick_example

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
    pick = pick_example(causal, ARTIFACTS.parent)
    if pick is None:
        print("no example case: run the causal evaluation first")
        return 1
    with connect() as conn:
        rows = conn.execute(_OFFICIAL_SQL, (pick["ev_id"],)).fetchall()
    out = {
        "ev_id": pick["ev_id"],
        "official": [{"label": label, "description": desc} for label, desc in rows],
        "source": "findings with cm_inPc = 'T' (scripts/export_readme_example.py)",
    }
    path = ARTIFACTS / EXAMPLE_OFFICIAL
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}: {len(rows)} official findings for {pick['ev_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
