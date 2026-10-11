"""Gold labels, the date split and migrations."""

import random
from datetime import date
from typing import Any

import pytest
from conftest import FixtureData

from causetrace.db import Connection
from causetrace.gold import get_gold_labels
from causetrace.migrate import apply_migrations, migration_files
from causetrace.split import HELD_OUT_FROM, load_split, split_events


def test_gold_labels_match_findings_in_pc(
    loaded: tuple[Connection, dict[str, Any]], fixture_data: FixtureData
) -> None:
    conn, _ = loaded
    taxonomy = {row[0] for row in conn.execute("SELECT code FROM labels").fetchall()}
    for ev_id in fixture_data.ev_ids:
        labels = get_gold_labels(ev_id, conn)
        assert labels == sorted(fixture_data.label_codes(ev_id, in_pc_only=True)), ev_id
        assert labels, ev_id
        assert set(labels) <= taxonomy


def test_label_short_names(loaded: tuple[Connection, dict[str, Any]]) -> None:
    conn, _ = loaded
    names: dict[str, str] = dict(conn.execute("SELECT code, short_name FROM labels").fetchall())
    assert names["02-04-15"] == "Personnel issues-Action/decision-Info processing/decision"
    # The full name wins over a truncated copy in older rows.
    assert names["02-02-20"] == "Personnel issues-Psychological-Perception/orientation/illusion"


def test_fixture_split(loaded: tuple[Connection, dict[str, Any]]) -> None:
    conn, _ = loaded
    split = load_split(conn)
    assert (len(split.historical), len(split.held_out)) == (200, 100)
    assert load_split(conn) == split
    assert not set(split.historical) & set(split.held_out)
    dates: dict[str, date] = dict(conn.execute("SELECT ev_id, ev_date FROM events").fetchall())
    assert all(dates[e] < HELD_OUT_FROM for e in split.historical)
    assert all(dates[e] >= HELD_OUT_FROM for e in split.held_out)
    ordered = [(dates[e], e) for e in split.historical]
    assert ordered == sorted(ordered)


def test_split_is_deterministic_and_order_independent() -> None:
    rng = random.Random(0)
    events = [
        (f"2000{i:04d}X{i:05d}", date(2015 + i % 12, 1 + i % 12, 1 + i % 28)) for i in range(500)
    ]
    expected = split_events(events)
    for _ in range(5):
        shuffled = events[:]
        rng.shuffle(shuffled)
        assert split_events(shuffled) == expected


def test_split_boundary() -> None:
    split = split_events([("B", date(2024, 1, 1)), ("A", date(2023, 12, 31))])
    assert split.historical == ("A",)
    assert split.held_out == ("B",)


def test_split_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError):
        split_events([("A", date(2020, 1, 1)), ("A", date(2025, 1, 1))])


def test_migrations_are_idempotent(db: Connection) -> None:
    assert apply_migrations(db) == []
    versions = [v for v, _ in migration_files()]
    applied = [r[0] for r in db.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    assert applied == versions


def test_changed_migration_is_an_error(db: Connection) -> None:
    db.execute("UPDATE schema_migrations SET checksum = 'tampered'")
    db.commit()
    with pytest.raises(RuntimeError, match="changed after it was applied"):
        apply_migrations(db)
