"""Deterministic stand-ins for the retrieval models (no downloads, no network)."""

import hashlib
import math
import re
from collections.abc import Sequence

DIM = 384
_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag of words, L2-normalized: texts sharing words get similar vectors."""

    model_name = "fake/hashing-embedder"
    revision = "v1"
    dim = DIM

    def __init__(self) -> None:
        self.calls = 0
        self.texts_embedded = 0

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in _WORD.findall(text.lower()):
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big")
            vec[h % DIM] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            vec[0], norm = 1.0, 1.0
        return [x / norm for x in vec]


class FakeReranker:
    """Scores a passage by how many distinct query words it contains."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls += 1
        q = set(_WORD.findall(query.lower()))
        return [float(len(q & set(_WORD.findall(p.lower())))) for p in passages]
