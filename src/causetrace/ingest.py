"""Load the NTSB ``mdb-export`` CSVs into PostgreSQL.

Reads events.csv, aircraft.csv, narratives.csv and Findings.csv from a directory,
validates every row with Pydantic, applies the narrative leakage gate, upserts
idempotently in one transaction and writes a JSON report.

Usage: python -m causetrace.ingest --data DIR [--report artifacts/ingest_report.json]
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from psycopg import sql
from pydantic import BaseModel, ValidationError

from causetrace.db import Connection, connect
from causetrace.leakage import sanitize_factual_narrative
from causetrace.migrate import apply_migrations
from causetrace.models import AircraftRow, EventRow, FindingRow, NarrativeRow

DEFAULT_REPORT = Path("artifacts/ingest_report.json")
MAX_EXAMPLES = 5

csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class TableSpec:
    table: str
    filename: str
    model: type[BaseModel]
    key: tuple[str, ...]
    parent_key: tuple[str, ...] | None  # columns that must match an accepted parent row


EVENTS = TableSpec("events", "events.csv", EventRow, ("ev_id",), None)
AIRCRAFT = TableSpec("aircraft", "aircraft.csv", AircraftRow, ("ev_id", "aircraft_key"), ("ev_id",))
NARRATIVES = TableSpec(
    "narratives",
    "narratives.csv",
    NarrativeRow,
    ("ev_id", "aircraft_key"),
    ("ev_id", "aircraft_key"),
)
FINDINGS = TableSpec(
    "findings",
    "Findings.csv",
    FindingRow,
    ("ev_id", "aircraft_key", "finding_no"),
    ("ev_id", "aircraft_key"),
)
SPECS = (EVENTS, AIRCRAFT, NARRATIVES, FINDINGS)
DB_TABLES = ("events", "aircraft", "narratives", "labels", "findings")


@dataclass
class TableResult:
    rows_read: int = 0
    accepted: list[Any] = field(default_factory=list)
    rejected: int = 0
    duplicates: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def reject(self, row_number: int, raw: dict[str, str], reason: str) -> None:
        self.rejected += 1
        self.reasons[reason] += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append({"row": row_number, "ev_id": raw.get("ev_id"), "reason": reason})

    def summary(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read,
            "accepted": len(self.accepted),
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "rejection_reasons": dict(sorted(self.reasons.items())),
            "rejection_examples": self.examples,
        }


def read_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _reason(err: ValidationError) -> str:
    e = err.errors()[0]
    loc = ".".join(str(part) for part in e["loc"]) or "row"
    return f"{loc}: {e['type']}"


def validate_table(
    spec: TableSpec, path: Path, parent_keys: set[tuple[Any, ...]] | None
) -> TableResult:
    """Validate rows; keep the first row per primary key; reject rows without a parent."""
    result = TableResult()
    seen: set[tuple[Any, ...]] = set()
    for row_number, raw in enumerate(read_csv(path), start=1):
        result.rows_read += 1
        try:
            row = spec.model.model_validate(raw)
        except ValidationError as err:
            result.reject(row_number, raw, _reason(err))
            continue
        key = tuple(getattr(row, k) for k in spec.key)
        if key in seen:
            result.duplicates += 1
            continue
        if (
            spec.parent_key is not None
            and parent_keys is not None
            and tuple(getattr(row, k) for k in spec.parent_key) not in parent_keys
        ):
            result.reject(row_number, raw, "orphan: no accepted parent row")
            continue
        seen.add(key)
        result.accepted.append(row)
    return result


def label_taxonomy(findings: list[FindingRow]) -> dict[str, str]:
    """Code -> short name. Old exports truncate descriptions, so keep the longest name."""
    names: dict[str, str] = {}
    for f in findings:
        name = f.label_short_name
        if name is None:
            continue
        best = names.get(f.label_code)
        if best is None or (len(name), best) > (len(best), name):
            names[f.label_code] = name
    for f in findings:
        names.setdefault(f.label_code, f.label_code)
    return dict(sorted(names.items()))


def _upsert(conn: Connection, table: str, key: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    updates = [c for c in cols if c not in key]
    query = sql.SQL(
        "INSERT INTO {table} ({cols}) VALUES ({vals}) ON CONFLICT ({key}) DO UPDATE SET {sets}"
    ).format(
        table=sql.Identifier(table),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        vals=sql.SQL(", ").join(sql.Placeholder() * len(cols)),
        key=sql.SQL(", ").join(map(sql.Identifier, key)),
        sets=sql.SQL(", ").join(
            sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c)) for c in updates
        ),
    )
    with conn.cursor() as cur:
        cur.executemany(query, [tuple(r[c] for c in cols) for r in rows])


def table_counts(conn: Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in DB_TABLES:
        row = conn.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
        ).fetchone()
        counts[table] = int(row[0]) if row else 0
    return counts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(conn: Connection, data_dir: Path) -> dict[str, Any]:
    """Validate and upsert ``data_dir``'s CSVs; return the ingest report."""
    for spec in SPECS:
        if not (data_dir / spec.filename).is_file():
            raise FileNotFoundError(f"missing {spec.filename} in {data_dir}")

    events = validate_table(EVENTS, data_dir / EVENTS.filename, None)
    event_keys = {(e.ev_id,) for e in events.accepted}
    aircraft = validate_table(AIRCRAFT, data_dir / AIRCRAFT.filename, event_keys)
    aircraft_keys = {(a.ev_id, a.aircraft_key) for a in aircraft.accepted}
    narratives = validate_table(NARRATIVES, data_dir / NARRATIVES.filename, aircraft_keys)
    findings = validate_table(FINDINGS, data_dir / FINDINGS.filename, aircraft_keys)
    results = {
        "events": events,
        "aircraft": aircraft,
        "narratives": narratives,
        "findings": findings,
    }

    # Leakage gate: gold text is pooled over all aircraft of the event.
    gold_by_event: dict[str, list[str | None]] = {}
    for n in narratives.accepted:
        gold_by_event.setdefault(n.ev_id, []).extend([n.narr_accf, n.narr_cause])
    narrative_rows: list[dict[str, Any]] = []
    sentences_total = sentences_removed = withheld = 0
    for n in narratives.accepted:
        gate = sanitize_factual_narrative(n.narr_accp, gold_by_event[n.ev_id])
        narrative_rows.append(
            n.model_dump()
            | {
                "narr_accp_safe": gate.text,
                "accp_sentences_total": gate.sentences_total,
                "accp_sentences_removed": gate.sentences_removed,
            }
        )
        sentences_total += gate.sentences_total
        sentences_removed += gate.sentences_removed
        withheld += gate.text is None

    labels = label_taxonomy(findings.accepted)
    with conn.transaction():
        apply_migrations(conn)
        _upsert(conn, EVENTS.table, EVENTS.key, [e.model_dump() for e in events.accepted])
        _upsert(conn, AIRCRAFT.table, AIRCRAFT.key, [a.model_dump() for a in aircraft.accepted])
        _upsert(conn, NARRATIVES.table, NARRATIVES.key, narrative_rows)
        label_rows = [{"code": c, "short_name": s} for c, s in labels.items()]
        _upsert(conn, "labels", ("code",), label_rows)
        _upsert(conn, FINDINGS.table, FINDINGS.key, [f.model_dump() for f in findings.accepted])
        db_counts = table_counts(conn)

    return {
        "source_dir": str(data_dir),
        "files": {spec.filename: {"sha256": _sha256(data_dir / spec.filename)} for spec in SPECS},
        "tables": {name: r.summary() for name, r in results.items()},
        "labels": {"taxonomy_size": len(labels)},
        "narrative_sanitization": {
            "narratives": len(narrative_rows),
            "withheld": withheld,
            "sentences_total": sentences_total,
            "sentences_removed": sentences_removed,
        },
        "database_counts": db_counts,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest NTSB mdb-export CSVs.")
    parser.add_argument("--data", type=Path, required=True, help="directory with the CSVs")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    with connect(args.database_url) as conn:
        report = ingest(conn, args.data)
    write_report(report, args.report)
    for name, t in report["tables"].items():
        print(
            f"{name}: read {t['rows_read']}, accepted {t['accepted']}, "
            f"rejected {t['rejected']}, duplicates {t['duplicates']}"
        )
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
