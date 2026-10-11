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


_ALL_GOLD_SQL = """
SELECT ev_id, array_agg(DISTINCT label_code ORDER BY label_code)
FROM findings
WHERE cm_inpc IS TRUE
GROUP BY ev_id
"""


def get_all_gold_labels(conn: Connection | None = None) -> dict[str, frozenset[str]]:
    """Gold label sets of every event that has any (events without labels are absent)."""
    if conn is None:
        with connect() as own:
            return get_all_gold_labels(own)
    return {ev_id: frozenset(codes) for ev_id, codes in conn.execute(_ALL_GOLD_SQL).fetchall()}
