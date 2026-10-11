"""Gold labels for scoring. GOLD_ONLY: never use these to build model input.

``causetrace.packets`` must never import this module (docs/DATA_LEAKAGE.md).
"""

from causetrace.db import Connection, connect

_GOLD_SQL = """
SELECT DISTINCT label_code
FROM findings
WHERE ev_id = %s AND cm_inpc IS TRUE
ORDER BY label_code
"""


def get_gold_labels(ev_id: str, conn: Connection | None = None) -> list[str]:
    """Sorted distinct ``category-subcategory-section`` codes of findings with cm_inPc = 'T'."""
    if conn is None:
        with connect() as own:
            return get_gold_labels(ev_id, own)
    return [row[0] for row in conn.execute(_GOLD_SQL, (ev_id,)).fetchall()]
