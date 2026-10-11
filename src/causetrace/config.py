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


# Retrieval models. Revisions are Hugging Face git refs; pin to a commit SHA for
# reproducible indexes (scripts/download_models.py prints the resolved SHA).
EMBEDDING_MODEL = os.environ.get("CAUSETRACE_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = os.environ.get("CAUSETRACE_EMBEDDING_REVISION") or "main"
EMBEDDING_DIM = 384  # fixed by the passages.embedding vector(384) column
RERANK_MODEL = os.environ.get("CAUSETRACE_RERANK_MODEL") or "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = os.environ.get("CAUSETRACE_RERANK_REVISION") or "main"
