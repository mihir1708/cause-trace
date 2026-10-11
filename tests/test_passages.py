"""Passage chunking and building."""

import itertools
from typing import Any

import pytest
from conftest import FixtureData

from causetrace.db import Connection
from causetrace.passages import NTSB_HEADINGS, build_passages, chunk_words, split_glued_headings


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
    assert counts["probable_cause"] == 300
    assert counts["analysis"] > 300  # long analyses are chunked like factual narratives
    assert counts["factual"] >= 300
    rows = conn.execute(
        "SELECT event_id, section_type, count(*), max(chunk_index) FROM passages GROUP BY 1, 2"
    ).fetchall()
    assert len({r[0] for r in rows}) == 300
    for _, section, n, max_chunk in rows:
        assert max_chunk == n - 1
        if section == "probable_cause":
            assert n == 1
    # Chunked sections never exceed the single-chunk threshold.
    longest = conn.execute(
        "SELECT section_type, max(array_length(regexp_split_to_array(btrim(content), '\\s+'), 1))"
        " FROM passages WHERE section_type IN ('factual', 'analysis') GROUP BY 1"
    ).fetchall()
    assert len(longest) == 2 and all(n <= 400 for _, n in longest)
    multi = conn.execute(
        "SELECT count(*) FROM passages WHERE section_type = 'analysis' AND chunk_index > 0"
    ).fetchone()
    assert multi is not None and multi[0] > 0
    # No heading stays glued to the next word.
    glued = conn.execute(
        "SELECT count(*) FROM passages WHERE content ~ '(HISTORY OF FLIGHT|INFORMATION)[A-Za-z0-9]'"
    ).fetchone()
    assert glued == (0,)
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HISTORY OF FLIGHTOn May 1, the pilot", "HISTORY OF FLIGHT On May 1, the pilot"),
        (
            "WRECKAGE AND IMPACT INFORMATIONThe wreckage",
            "WRECKAGE AND IMPACT INFORMATION The wreckage",
        ),
        ("AIRPORT INFORMATIONhe airport", "AIRPORT INFORMATION he airport"),
        ("TESTS AND RESEARCHGlassy residue", "TESTS AND RESEARCH Glassy residue"),
        (
            "at 1200 AKDT WRECKAGE AND IMPACT INFORMATIONOn",
            "at 1200 AKDT WRECKAGE AND IMPACT INFORMATION On",
        ),
        ("COMMUNICATIONSAccording to ATC", "COMMUNICATIONS According to ATC"),
        ("PERSONNEL INFORMATIONA review of", "PERSONNEL INFORMATION A review of"),
        ("AIRPORT INFORMATIONN93AA, serial", "AIRPORT INFORMATION N93AA, serial"),
        (
            "ORGANIZATIONAL AND MANAGEMENT INFORMATIONMIE",
            "ORGANIZATIONAL AND MANAGEMENT INFORMATION MIE",
        ),
        ("FIREThe fire", "FIRE The fire"),
        # Already separated, longer uppercase words, acronyms and mid-word matches stay.
        ("HISTORY OF FLIGHT\nOn May 1", "HISTORY OF FLIGHT\nOn May 1"),
        ("FIREFIGHTING crews responded", "FIREFIGHTING crews responded"),
        ("PIREPs, NOTAMs and AIRMETs", "PIREPs, NOTAMs and AIRMETs"),
        ("SPITFIREThe", "SPITFIREThe"),
        ("the fire started", "the fire started"),
    ],
)
def test_split_glued_headings(raw: str, expected: str) -> None:
    assert split_glued_headings(raw) == expected


def test_every_heading_is_split() -> None:
    for heading in NTSB_HEADINGS:
        assert split_glued_headings(f"{heading}The text") == f"{heading} The text"
