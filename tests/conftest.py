"""Shared fixtures. Tests need PostgreSQL at TEST_DATABASE_URL and nothing else."""

import csv
import socket
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from causetrace.config import test_database_url
from causetrace.db import Connection
from causetrace.ingest import ingest
from causetrace.migrate import apply_migrations

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
TABLES = (
    "analysis_runs",
    "embedding_cache",
    "lexeme_stats",
    "passages",
    "findings",
    "labels",
    "narratives",
    "aircraft",
    "events",
    "schema_migrations",
)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

csv.field_size_limit(sys.maxsize)


@pytest.fixture(autouse=True, scope="session")
def _no_external_network() -> Iterator[None]:
    """Fail any Python-level connection to a non-loopback host (CLAUDE.md rule 4)."""
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in _LOCAL_HOSTS:
            raise RuntimeError(f"tests must not use the network (tried {address!r})")
        real_connect(self, address)

    socket.socket.connect = guarded  # type: ignore[method-assign,assignment]
    yield
    socket.socket.connect = real_connect  # type: ignore[method-assign]


@pytest.fixture(scope="session")
def db_url() -> str:
    """The test database URL; the database is created if it does not exist."""
    url = test_database_url()
    try:
        psycopg.connect(url).close()
    except psycopg.OperationalError as err:
        if "does not exist" not in str(err):
            pytest.fail(f"cannot reach PostgreSQL at TEST_DATABASE_URL ({err}); run `make up`")
        admin = make_conninfo(url, dbname="postgres")
        with psycopg.connect(admin, autocommit=True) as conn:
            name = conninfo_to_dict(url)["dbname"]
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(str(name))))
    return url


def reset(conn: Connection) -> None:
    """Drop all CauseTrace tables and re-apply migrations."""
    conn.execute(
        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
            sql.SQL(", ").join(map(sql.Identifier, TABLES))
        )
    )
    conn.commit()
    apply_migrations(conn)


@pytest.fixture
def db(db_url: str) -> Iterator[Connection]:
    """A connection to a freshly reset, migrated, empty test database."""
    with psycopg.connect(db_url, autocommit=True) as conn:
        reset(conn)
        yield conn


@pytest.fixture(scope="module")
def loaded(db_url: str) -> Iterator[tuple[Connection, dict[str, Any]]]:
    """A test database with the 300-event fixture ingested once per module."""
    with psycopg.connect(db_url, autocommit=True) as conn:
        reset(conn)
        report = ingest(conn, FIXTURES)
        yield conn, report


@dataclass(frozen=True)
class FixtureData:
    """The fixture CSVs read directly, independent of the package's ingest code."""

    events: list[dict[str, str]]
    aircraft: list[dict[str, str]]
    narratives: list[dict[str, str]]
    findings: list[dict[str, str]]

    @property
    def ev_ids(self) -> list[str]:
        return [e["ev_id"] for e in self.events]

    def gold_texts(self, ev_id: str) -> list[str]:
        return [
            n[col].strip()
            for n in self.narratives
            if n["ev_id"] == ev_id
            for col in ("narr_accf", "narr_cause")
            if n[col].strip()
        ]

    def label_codes(self, ev_id: str, in_pc_only: bool = False) -> set[str]:
        return {
            f"{f['category_no']}-{f['subcategory_no']}-{f['section_no']}"
            for f in self.findings
            if f["ev_id"] == ev_id and (f["cm_inPc"] == "T" or not in_pc_only)
        }

    def finding_codes(self, ev_id: str) -> set[str]:
        return {f["finding_code"] for f in self.findings if f["ev_id"] == ev_id}


def _read(name: str) -> list[dict[str, str]]:
    with (FIXTURES / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="session")
def fixture_data() -> FixtureData:
    return FixtureData(
        events=_read("events.csv"),
        aircraft=_read("aircraft.csv"),
        narratives=_read("narratives.csv"),
        findings=_read("Findings.csv"),
    )
