"""The label taxonomy the agent may predict from.

Codes that occur as ``cm_inPc = 'T'`` findings of history-split events only, so the
list of possible answers is built from earlier investigations and held-out events cannot
add a code to it (docs/DATA_LEAKAGE.md, "Retrieval context").
"""

import hashlib

from causetrace.db import Connection
from causetrace.split import HELD_OUT_FROM

_TAXONOMY_SQL = """
SELECT l.code, l.short_name
FROM labels l
WHERE EXISTS (
    SELECT 1 FROM findings f JOIN events e ON e.ev_id = f.ev_id
    WHERE f.label_code = l.code AND f.cm_inpc IS TRUE AND e.ev_date < %s
)
ORDER BY l.code
"""


def load_taxonomy(conn: Connection) -> dict[str, str]:
    """Code -> short name, sorted by code."""
    return {code: name for code, name in conn.execute(_TAXONOMY_SQL, (HELD_OUT_FROM,))}


def render_taxonomy(taxonomy: dict[str, str]) -> str:
    lines = [f"{code}  {name}" for code, name in taxonomy.items()]
    return f"Label taxonomy ({len(lines)} codes; use only these):\n\n" + "\n".join(lines)


def taxonomy_sha256(taxonomy: dict[str, str]) -> str:
    return hashlib.sha256(render_taxonomy(taxonomy).encode("utf-8")).hexdigest()
