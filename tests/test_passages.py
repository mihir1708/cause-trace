"""Passage chunking and building."""

import itertools
from typing import Any

import pytest
from conftest import FixtureData

from causetrace.db import Connection
from causetrace.passages import build_passages, chunk_words


def _text(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize("n", [0, 1, 300, 400])
def test_short_texts_stay_whole(n: int) -> None:
    chunks = chunk_words(_text(n))
    assert chunks == ([_text(n)] if n else [])


@pytest.mark.parametrize("n", [401, 550, 1000, 2345])
def test_long_texts_chunk_with_exact_overlap(n: int) -> None:
    chunks = [c.split() for c in chunk_words(_text(n))]
    assert len(chunks) >= 2
    assert all(len(c) <= 300 for c in chunks)
    assert all(len(c) >= 200 for c in chunks)  # no short tail
    assert chunks[0][0] == "w0" and chunks[-1][-1] == f"w{n - 1}"
    for a, b in itertools.pairwise(chunks):
        assert a[-50:] == b[:50]
    covered = {w for c in chunks for w in c}
    assert covered == set(_text(n).split())


def test_build_passages_on_fixture(
    loaded: tuple[Connection, dict[str, Any]], fixture_data: FixtureData
) -> None:
    conn, _ = loaded
    counts = build_passages(conn)
    assert counts["analysis"] == counts["probable_cause"] == 300
    assert counts["factual"] >= 300
    rows = conn.execute(
        "SELECT event_id, section_type, count(*), max(chunk_index) FROM passages GROUP BY 1, 2"
    ).fetchall()
    assert len({r[0] for r in rows}) == 300
    for _, section, n, max_chunk in rows:
        assert max_chunk == n - 1
        if section != "factual":
            assert n == 1
    # Metadata comes from the event and its first aircraft.
    row = conn.execute(
        "SELECT event_date, aircraft_category FROM passages"
        " WHERE passage_id = '20080219X00204:probable_cause:0'"
    ).fetchone()
    assert row is not None and row[0].isoformat() == "2008-01-11" and row[1] == "AIR"
    # Rebuilding is idempotent.
    before = conn.execute("SELECT count(*) FROM passages").fetchone()
    assert build_passages(conn) == counts
    assert conn.execute("SELECT count(*) FROM passages").fetchone() == before
