"""Ingestion: validation, idempotent upserts, report."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import FIXTURES

from causetrace.db import Connection
from causetrace.ingest import ingest, table_counts, write_report
from causetrace.models import EventRow, parse_mdb_datetime


def test_fixture_ingest_loads_all_events(db: Connection) -> None:
    report = ingest(db, FIXTURES)
    tables = report["tables"]
    assert tables["events"]["rows_read"] == tables["events"]["accepted"] == 300
    for name in ("events", "aircraft", "narratives", "findings"):
        assert tables[name]["rejected"] == 0, tables[name]
        assert tables[name]["duplicates"] == 0
    assert report["database_counts"]["events"] == 300
    assert report["database_counts"]["aircraft"] == 303
    assert report["database_counts"]["narratives"] == 303
    assert report["database_counts"]["findings"] == 972
    assert report["labels"]["taxonomy_size"] == report["database_counts"]["labels"] > 0


def test_ingesting_twice_gives_identical_counts(db: Connection, tmp_path: Path) -> None:
    first = ingest(db, FIXTURES)
    counts_after_first = table_counts(db)
    second = ingest(db, FIXTURES)
    assert table_counts(db) == counts_after_first
    assert second == first
    write_report(first, tmp_path / "a.json")
    write_report(second, tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def _subset(src: str, dst: Path, rows: list[dict[str, str]]) -> None:
    with (FIXTURES / src).open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def _rows(name: str, ev_id: str) -> list[dict[str, str]]:
    with (FIXTURES / name).open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["ev_id"] == ev_id]


def test_invalid_duplicate_and_orphan_rows(db: Connection, tmp_path: Path) -> None:
    ev_id = "20080219X00204"
    event = _rows("events.csv", ev_id)[0]
    bad_date = event | {"ev_id": "20080219X00999", "ev_date": "2008-01-11"}
    wrong_year = event | {"ev_id": "20080219X00998", "ev_year": "2009"}
    _subset("events.csv", tmp_path / "events.csv", [event, event, bad_date, wrong_year])

    aircraft = _rows("aircraft.csv", ev_id)
    orphan = aircraft[0] | {"ev_id": "20080219X00999"}
    _subset("aircraft.csv", tmp_path / "aircraft.csv", [*aircraft, orphan])
    _subset("narratives.csv", tmp_path / "narratives.csv", _rows("narratives.csv", ev_id))

    findings = _rows("Findings.csv", ev_id)
    bad_category = findings[0] | {"finding_no": "99", "category_no": "2"}
    _subset("Findings.csv", tmp_path / "Findings.csv", [*findings, bad_category])

    report = ingest(db, tmp_path)
    ev = report["tables"]["events"]
    assert (ev["rows_read"], ev["accepted"], ev["rejected"], ev["duplicates"]) == (4, 1, 2, 1)
    assert ev["rejection_reasons"] == {"ev_date: value_error": 1, "row: value_error": 1}
    ac = report["tables"]["aircraft"]
    assert (ac["accepted"], ac["rejected"]) == (1, 1)
    assert ac["rejection_reasons"] == {"orphan: no accepted parent row": 1}
    fi = report["tables"]["findings"]
    assert (fi["accepted"], fi["rejected"]) == (len(findings), 1)
    assert report["database_counts"]["events"] == 1


def test_missing_file_is_an_error(db: Connection, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest(db, tmp_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10/01/26 07:36:00", datetime(2026, 10, 1, 7, 36)),
        ("01/10/08 00:00:00", datetime(2008, 1, 10)),
        ("05/05/62 12:00:00", datetime(1962, 5, 5, 12)),
        ("12/31/2023", datetime(2023, 12, 31)),
    ],
)
def test_parse_mdb_datetime(raw: str, expected: datetime) -> None:
    assert parse_mdb_datetime(raw) == expected


@pytest.mark.parametrize("raw", ["2026-10-01", "13/01/26 00:00:00", ""])
def test_parse_mdb_datetime_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_mdb_datetime(raw)


def test_event_row_blank_cells_are_none() -> None:
    row: dict[str, Any] = {"ev_id": "20080219X00204", "ev_date": "01/11/08 00:00:00"}
    event = EventRow.model_validate(row | {"ev_city": "  ", "wx_temp": ""})
    assert event.ev_city is None and event.wx_temp is None


def test_cli_writes_report(db: Connection, db_url: str, tmp_path: Path) -> None:
    from causetrace.ingest import main

    report_path = tmp_path / "report.json"
    main(["--data", str(FIXTURES), "--report", str(report_path), "--database-url", db_url])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["tables"]["events"]["accepted"] == 300
