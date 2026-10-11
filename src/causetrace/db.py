"""PostgreSQL connections."""

from typing import Any

import psycopg

from causetrace.config import database_url

Connection = psycopg.Connection[tuple[Any, ...]]


def connect(url: str | None = None) -> Connection:
    """Open a connection to ``url`` (default: ``DATABASE_URL``)."""
    return psycopg.connect(url or database_url())
