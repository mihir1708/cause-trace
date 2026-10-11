"""Runtime configuration from environment variables."""

import os

DEFAULT_DATABASE_URL = "postgresql://causetrace:causetrace@localhost:5432/causetrace"
DEFAULT_TEST_DATABASE_URL = "postgresql://causetrace:causetrace@localhost:5432/causetrace_test"


def database_url() -> str:
    """Database for migrate/ingest/packets: ``DATABASE_URL``."""
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def test_database_url() -> str:
    """Database the test suite owns and resets: ``TEST_DATABASE_URL``."""
    return os.environ.get("TEST_DATABASE_URL") or DEFAULT_TEST_DATABASE_URL
