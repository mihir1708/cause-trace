"""Store and read agent runs (``analysis_runs``)."""

import json
from typing import Any
from uuid import UUID

from causetrace.db import Connection

_COLUMNS = (
    "run_id, created_at, ev_id, status, model, abstained, grounding_score, request, result, error"
)


def save_run(
    conn: Connection,
    run_id: UUID,
    *,
    ev_id: str | None,
    status: str,
    request: dict[str, Any],
    model: str | None = None,
    result: dict[str, Any] | None = None,
    abstained: bool | None = None,
    grounding_score: float | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO analysis_runs
           (run_id, ev_id, status, model, abstained, grounding_score, request, result, error)
           VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)""",
        (
            run_id,
            ev_id,
            status,
            model,
            abstained,
            grounding_score,
            json.dumps(request, default=str),
            None if result is None else json.dumps(result, default=str),
            error,
        ),
    )


def get_run(conn: Connection, run_id: UUID) -> dict[str, Any] | None:
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM analysis_runs WHERE run_id = %s", (run_id,)
    ).fetchone()
    if row is None:
        return None
    return dict(zip(_COLUMNS.split(", "), row, strict=True))
