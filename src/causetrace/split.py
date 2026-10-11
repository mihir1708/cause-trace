"""Deterministic date-based split (docs/EVAL_PROTOCOL.md).

Held-out = ev_date >= 2024-01-01, history = earlier. Lists are ordered by (ev_date, ev_id).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from causetrace.db import Connection, connect

HELD_OUT_FROM = date(2024, 1, 1)


@dataclass(frozen=True)
class Split:
    historical: tuple[str, ...]
    held_out: tuple[str, ...]


def split_events(events: Iterable[tuple[str, date]], cutoff: date = HELD_OUT_FROM) -> Split:
    """Split ``(ev_id, ev_date)`` pairs; the result does not depend on input order."""
    ordered = sorted(events, key=lambda e: (e[1], e[0]))
    ids = [ev_id for ev_id, _ in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ev_id in split input")
    return Split(
        historical=tuple(ev_id for ev_id, d in ordered if d < cutoff),
        held_out=tuple(ev_id for ev_id, d in ordered if d >= cutoff),
    )


def load_split(conn: Connection | None = None) -> Split:
    """Split every event in the database."""
    if conn is None:
        with connect() as own:
            return load_split(own)
    rows = conn.execute("SELECT ev_id, ev_date FROM events").fetchall()
    return split_events((r[0], r[1]) for r in rows)
