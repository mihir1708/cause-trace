"""Dense embeddings: the embedder interface, the sentence-transformers implementation and
the resumable, cached embedding of passages.

sentence-transformers and torch are in the optional ``ml`` dependency group and are
imported lazily, so the rest of the package works without them.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from causetrace.config import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_REVISION
from causetrace.db import Connection

DEFAULT_BATCH_SIZE = 64


class Embedder(Protocol):
    """Maps texts to L2-normalized vectors of ``dim`` floats."""

    model_name: str
    revision: str
    dim: int

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]: ...


def select_device() -> str:
    """``cuda`` or ``mps`` when available, otherwise ``cpu``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEmbedder:
    """``BAAI/bge-small-en-v1.5`` (by default) via sentence-transformers.

    Queries are whole narratives compared with passages, so no query instruction is
    prepended by default (BGE's instruction is meant for short questions).
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        revision: str = EMBEDDING_REVISION,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        query_instruction: str = "",
        local_files_only: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.device = device or select_device()
        self._model = SentenceTransformer(
            model_name, revision=revision, device=self.device, local_files_only=local_files_only
        )
        model: Any = self._model
        # Renamed in sentence-transformers 6; keep working with older versions.
        get_dim = getattr(model, "get_embedding_dimension", None)
        dim = get_dim() if get_dim else model.get_sentence_embedding_dimension()
        if dim != EMBEDDING_DIM:
            raise ValueError(f"{model_name} has {dim} dimensions, the schema needs {EMBEDDING_DIM}")
        self.dim = EMBEDDING_DIM

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        inputs = [self.query_instruction + t for t in texts] if is_query else list(texts)
        vectors: Any = self._model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors.tolist()]


@lru_cache(maxsize=1)
def default_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def to_pgvector(vector: Sequence[float]) -> str:
    """pgvector text literal, e.g. ``[0.1,0.2]``."""
    return "[" + ",".join(format(x, ".8g") for x in vector) + "]"


@dataclass(frozen=True)
class EmbedStats:
    from_cache: int  # passages filled from embedding_cache
    embedded_texts: int  # distinct texts sent to the model
    remaining: int  # passages still without a current embedding (0 when complete)


_STALE = """(p.embedding IS NULL
     OR p.embedding_model IS DISTINCT FROM %(model)s
     OR p.embedding_revision IS DISTINCT FROM %(revision)s)"""

_FILL_FROM_CACHE = f"""
UPDATE passages p
SET embedding = c.embedding, embedding_model = c.model, embedding_revision = c.revision
FROM embedding_cache c
WHERE c.model = %(model)s AND c.revision = %(revision)s
  AND c.content_sha256 = p.content_sha256 AND {_STALE}
"""

_NEXT_BATCH = f"""
SELECT DISTINCT ON (p.content_sha256) p.content_sha256, p.content
FROM passages p
WHERE {_STALE}
ORDER BY p.content_sha256
LIMIT %(limit)s
"""


def embed_passages(
    conn: Connection,
    embedder: Embedder,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> EmbedStats:
    """Give every passage an embedding from ``embedder``, committing after each batch.

    Cached embeddings (same model, revision and text) are reused, so an interrupted run
    resumes where it stopped. ``max_batches`` limits the work done in one call.
    """
    if embedder.dim != EMBEDDING_DIM:
        raise ValueError(f"embedder has {embedder.dim} dimensions, need {EMBEDDING_DIM}")
    params: dict[str, Any] = {"model": embedder.model_name, "revision": embedder.revision}
    from_cache = conn.execute(_FILL_FROM_CACHE, params).rowcount
    conn.commit()

    embedded = batches = 0
    while max_batches is None or batches < max_batches:
        rows = conn.execute(_NEXT_BATCH, params | {"limit": batch_size}).fetchall()
        if not rows:
            break
        vectors = embedder.embed([content for _, content in rows])
        if len(vectors) != len(rows) or any(len(v) != EMBEDDING_DIM for v in vectors):
            raise ValueError("embedder returned the wrong number or size of vectors")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embedding_cache (model, revision, content_sha256, embedding)"
                " VALUES (%s, %s, %s, %s::vector) ON CONFLICT DO NOTHING",
                [
                    (embedder.model_name, embedder.revision, sha, to_pgvector(vec))
                    for (sha, _), vec in zip(rows, vectors, strict=True)
                ],
            )
        conn.execute(_FILL_FROM_CACHE, params)
        conn.commit()
        embedded += len(rows)
        batches += 1
        if progress is not None:
            progress(embedded)

    remaining_row = conn.execute(f"SELECT count(*) FROM passages p WHERE {_STALE}", params)
    count = remaining_row.fetchone()
    return EmbedStats(
        from_cache=from_cache, embedded_texts=embedded, remaining=int(count[0]) if count else 0
    )
