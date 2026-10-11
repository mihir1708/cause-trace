"""Runtime configuration from environment variables (and ``.env`` for LLM settings)."""

import os
from dataclasses import dataclass
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
# Seconds per Claude request (the SDK default of 600 s let one hung call stall a run), and
# SDK retries per call (connection errors, 408/409/429/5xx); a timed-out request is retried.
LLM_TIMEOUT_S = float(os.environ.get("CAUSETRACE_LLM_TIMEOUT_S") or 120)
LLM_MAX_RETRIES = int(os.environ.get("CAUSETRACE_LLM_MAX_RETRIES") or 2)
# API database pool (causetrace.api): connections, and seconds a request waits for a free one
# before it is answered 503 "server busy" (searches hold a connection for their whole run).
API_DB_POOL_SIZE = int(os.environ.get("CAUSETRACE_API_DB_POOL_SIZE") or 20)
API_POOL_WAIT_S = float(os.environ.get("CAUSETRACE_API_POOL_WAIT_S") or 30)
# Free local fallback (causetrace.llm.ollama_provider).
OLLAMA_URL = os.environ.get("CAUSETRACE_OLLAMA_URL") or "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("CAUSETRACE_OLLAMA_MODEL") or "llama3.1:8b"


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


@dataclass(frozen=True)
class AgentLimits:
    """Hard limits of the multi-hop agent (causetrace.agent); the graph always terminates."""

    max_queries: int = 3  # planned search queries
    max_extra_hops: int = 1  # gap-query retrievals after check_evidence
    max_revisions: int = 1  # analyze -> verify -> analyze again
    max_passages: int = 10  # passages kept after merging all queries
    max_findings: int = 5  # labels in one answer
    max_tokens: int = 80_000  # LLM tokens per run (input + output + cache); checked per call
    rerank_max_words: int = 60  # shorter queries use rrf_rerank, longer ones rrf
    per_query_top_k: int = 10  # passages fetched per query before merging


AGENT_LIMITS = AgentLimits(
    max_tokens=int(os.environ.get("CAUSETRACE_AGENT_MAX_TOKENS") or AgentLimits.max_tokens)
)
