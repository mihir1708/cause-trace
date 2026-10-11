"""Runtime configuration from environment variables (and ``.env`` for LLM settings)."""

import os
from pathlib import Path

DEFAULT_DATABASE_URL = "postgresql://causetrace:causetrace@localhost:5432/causetrace"
DEFAULT_TEST_DATABASE_URL = "postgresql://causetrace:causetrace@localhost:5432/causetrace_test"


def database_url() -> str:
    """Database for migrate/ingest/packets: ``DATABASE_URL``."""
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def test_database_url() -> str:
    """Database the test suite owns and resets: ``TEST_DATABASE_URL``."""
    return os.environ.get("TEST_DATABASE_URL") or DEFAULT_TEST_DATABASE_URL


# Retrieval models, pinned to Hugging Face commit SHAs for reproducible indexes.
EMBEDDING_MODEL = os.environ.get("CAUSETRACE_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = (
    os.environ.get("CAUSETRACE_EMBEDDING_REVISION") or "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
EMBEDDING_DIM = 384  # fixed by the passages.embedding vector(384) column
RERANK_MODEL = os.environ.get("CAUSETRACE_RERANK_MODEL") or "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = (
    os.environ.get("CAUSETRACE_RERANK_REVISION") or "233902d25c440f23af6f7d6e94d2946bac0bee0a"
)


# LLM provider (causetrace.llm). The API key and budget come from the environment or .env.
LLM_MODEL = os.environ.get("CAUSETRACE_LLM_MODEL") or "claude-haiku-5-5"
DEFAULT_MAX_USD = 10.0


def load_dotenv(path: Path = Path(".env")) -> None:
    """Set variables from ``path`` (KEY=VALUE lines) that are not already in the environment."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def anthropic_api_key() -> str | None:
    load_dotenv()
    return os.environ.get("ANTHROPIC_API_KEY") or None


def max_usd() -> float:
    """Spend limit per process for LLM calls: ``MAX_USD`` (default 10)."""
    load_dotenv()
    return float(os.environ.get("MAX_USD") or DEFAULT_MAX_USD)
