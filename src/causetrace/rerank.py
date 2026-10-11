"""Cross-encoder reranking. sentence-transformers is imported lazily (``ml`` group)."""

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

from causetrace.config import RERANK_MODEL, RERANK_REVISION
from causetrace.embeddings import select_device

MAX_LENGTH = 512  # tokens per (query, passage) pair
QUERY_MAX_TOKENS = 192  # queries can be whole narratives; leave the rest for the passage


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """``cross-encoder/ms-marco-MiniLM-L6-v2`` (by default); higher score = more relevant."""

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        revision: str = RERANK_REVISION,
        *,
        device: str | None = None,
        batch_size: int = 32,
        local_files_only: bool = False,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self._model = CrossEncoder(
            model_name,
            revision=revision,
            device=device or select_device(),
            max_length=MAX_LENGTH,
            local_files_only=local_files_only,
        )

    def truncate_query(self, query: str) -> str:
        """Cut the query to QUERY_MAX_TOKENS tokens so passages are not crowded out."""
        tokenizer: Any = self._model.tokenizer
        ids = tokenizer(query, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) <= QUERY_MAX_TOKENS:
            return query
        return str(tokenizer.decode(ids[:QUERY_MAX_TOKENS]))

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        q = self.truncate_query(query)
        scores: Any = self._model.predict(
            [(q, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def default_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
