"""Hybrid passage retrieval: dense (pgvector HNSW), lexical (tsvector), RRF and reranking.

One entry point, ``search``; ``search_events`` collapses passage hits to events.
``before_date`` and ``exclude_event_id`` are applied in the SQL of every candidate query,
so no mode can return a passage from the target event or from on/after ``before_date``.
By default (``history_only=True``) ``before_date`` is also capped at HELD_OUT_FROM, so only
history passages can be returned; only an explicit ``history_only=False`` lifts the cap.
Passages contain GOLD_ONLY text of their own events (docs/DATA_LEAKAGE.md).

With ``rerank_fallback=True`` (the API) a reranker failure (model missing, runtime error)
degrades ``rrf_rerank`` to the RRF list and is reported in ``fallback``; by default it
raises, so an evaluation never scores RRF as if it were reranked. Every stage runs in an
OpenTelemetry span.
"""

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from opentelemetry.trace import Span, Status, StatusCode
from psycopg import sql

from causetrace.db import Connection, connect
from causetrace.embeddings import Embedder, default_embedder, to_pgvector
from causetrace.rerank import Reranker, default_reranker
from causetrace.split import HELD_OUT_FROM, corpus_cutoff
from causetrace.telemetry import tracer

log = logging.getLogger(__name__)

Mode = Literal["dense", "lexical", "rrf", "rrf_rerank"]
MODES: tuple[Mode, ...] = ("dense", "lexical", "rrf", "rrf_rerank")

CANDIDATES = 50  # depth of the dense and lexical lists, and of the fused list
RRF_K = 60
MAX_QUERY_TERMS = 30
HNSW_EF_SEARCH = 200  # > CANDIDATES so filtered HNSW scans still fill the list
# ts_rank_cd normalization: 1 (divide by 1 + log length) | 4 (mean harmonic extent distance).
# Without length normalization, long passages win OR queries built from whole narratives.
TS_RANK_NORMALIZATION = 1 | 4
# Parallel workers for the keyword query. Nearly every passage matches an OR query built from
# a narrative, so ts_rank_cd over ~100k passages dominates search time; more workers split
# that scan. The ORDER BY is a total order (score, passage_id), so the ranking is identical.
LEXICAL_PARALLEL_WORKERS = 4
LEXICAL_MIN_PARALLEL_SCAN = "1MB"  # lets the planner use all of them on a ~150 MB table


@dataclass(frozen=True)
class SearchHit:
    passage_id: str
    event_id: str
    score: float
    rank: int  # 1-based
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    fallback: str | None = None  # set when a stage failed and a simpler ranking was used


@dataclass(frozen=True)
class EventHit:
    event_id: str
    score: float  # score of the event's best passage
    rank: int  # 1-based
    hits: list[SearchHit]  # the event's passages in the ranked list, best first


@dataclass(frozen=True)
class EventSearchResult:
    events: list[EventHit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    fallback: str | None = None


def rrf_fuse(
    rankings: Sequence[Sequence[str]], k: int = RRF_K, limit: int = CANDIDATES
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(id) = sum over lists of 1 / (k + rank), rank from 1.

    Ties are broken by best single rank, then by id, so the result is deterministic.
    """
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best[item] = min(best.get(item, rank), rank)
    ordered = sorted(scores, key=lambda i: (-scores[i], best[i], i))
    return [(i, scores[i]) for i in ordered[:limit]]


def build_or_tsquery(lexemes: Sequence[str]) -> str | None:
    """A tsquery string ``'a' | 'b'`` matching any of the (already normalized) lexemes."""
    quoted = ["'" + lex.replace("\\", "\\\\").replace("'", "''") + "'" for lex in lexemes if lex]
    return " | ".join(quoted) or None


_QUERY_TERMS_SQL = """
SELECT u.lexeme, coalesce(array_length(u.positions, 1), 1) AS tf, s.ndoc
FROM unnest(to_tsvector('english', %(query)s)) AS u
JOIN lexeme_stats s ON s.lexeme = u.lexeme
"""


def history_passage_count(conn: Connection) -> int:
    """Number of historical-split passages (the N of the idf), at least 1."""
    n_row = conn.execute(
        "SELECT count(*) FROM passages WHERE event_date < %s", (HELD_OUT_FROM,)
    ).fetchone()
    return max(int(n_row[0]) if n_row else 0, 1)


def salient_terms(
    conn: Connection, query: str, limit: int = MAX_QUERY_TERMS, *, n_docs: int | None = None
) -> list[str]:
    """Up to ``limit`` query lexemes with the highest tf * idf over historical passages.

    Lexemes that never occur in the corpus cannot match and are dropped. ``n_docs`` is the
    history passage count when the caller has it (counted here otherwise).
    """
    if n_docs is None:
        n_docs = history_passage_count(conn)
    rows = conn.execute(_QUERY_TERMS_SQL, {"query": query}).fetchall()
    weighted = [(tf * math.log(1 + n_docs / ndoc), lex) for lex, tf, ndoc in rows]
    weighted.sort(key=lambda w: (-w[0], w[1]))
    return [lex for _, lex in weighted[:limit]]


def refresh_lexeme_stats(conn: Connection) -> int:
    """Recompute document frequencies over historical-split passages; return lexeme count.

    Only passages before HELD_OUT_FROM count, so held-out text never shapes term weights.
    """
    inner = sql.SQL("SELECT tsv FROM passages WHERE event_date < {}").format(
        sql.Literal(HELD_OUT_FROM)
    )
    with conn.transaction():
        conn.execute("DELETE FROM lexeme_stats")
        conn.execute(
            "INSERT INTO lexeme_stats (lexeme, ndoc) SELECT word, ndoc FROM ts_stat(%s)",
            (inner.as_string(conn),),
        )
    conn.commit()
    row = conn.execute("SELECT count(*) FROM lexeme_stats").fetchone()
    return int(row[0]) if row else 0


_COLUMNS = """passage_id, event_id, content, event_date, section_type, chunk_index,
       aircraft_category, phase_of_flight"""

# Both candidate queries carry the leakage filters; keep them in sync.
_FILTERS = """event_date < coalesce(%(before)s::date, 'infinity'::date)
  AND (%(exclude)s::text IS NULL OR event_id <> %(exclude)s::text)"""

_DENSE_SQL = f"""
SELECT {_COLUMNS}, 1 - (embedding <=> %(vec)s::vector) AS score
FROM passages
WHERE embedding IS NOT NULL AND {_FILTERS}
ORDER BY embedding <=> %(vec)s::vector
LIMIT %(limit)s
"""

_LEXICAL_SQL = f"""
SELECT {_COLUMNS}, ts_rank_cd(tsv, %(tsq)s::tsquery, %(norm)s) AS score
FROM passages
WHERE tsv @@ %(tsq)s::tsquery AND {_FILTERS}
ORDER BY score DESC, passage_id
LIMIT %(limit)s
"""


def _hit(row: tuple[Any, ...], score: float, rank: int) -> SearchHit:
    passage_id, event_id, content, event_date, section, chunk, category, phase = row[:8]
    return SearchHit(
        passage_id=passage_id,
        event_id=event_id,
        score=score,
        rank=rank,
        content=content,
        metadata={
            "event_date": event_date,
            "section_type": section,
            "chunk_index": chunk,
            "aircraft_category": category,
            "phase_of_flight": phase,
        },
    )


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class Retriever:
    """Runs searches against ``conn``; models are loaded on first use if not given.

    ``rerank_fallback``: on a reranker failure return the RRF ranking (and say so in
    ``fallback``) instead of raising.
    """

    def __init__(
        self,
        conn: Connection,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        *,
        rerank_fallback: bool = False,
    ) -> None:
        self.conn = conn
        self.rerank_fallback = rerank_fallback
        self._embedder = embedder
        self._reranker = reranker
        self._iterative_scan: bool | None = None
        self._n_docs: int | None = None  # history passage count, counted once per retriever

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = default_embedder()
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = default_reranker()
        return self._reranker

    def _supports_iterative_scan(self) -> bool:
        """pgvector >= 0.8 can keep scanning HNSW until filtered results fill LIMIT."""
        if self._iterative_scan is None:
            row = self.conn.execute(
                "SELECT string_to_array(extversion, '.')::int[] >= '{0,8}'"
                " FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            self._iterative_scan = bool(row and row[0])
        return self._iterative_scan

    def dense(
        self, query: str, before: date | None, exclude: str | None, timings: dict[str, float]
    ) -> list[SearchHit]:
        with tracer.start_as_current_span("retrieval.embed") as span:
            start = time.perf_counter()
            vector = self.embedder.embed([query], is_query=True)[0]
            timings["embed_ms"] = _ms(start)
            span.set_attribute("retrieval.embed_ms", timings["embed_ms"])
        with tracer.start_as_current_span("retrieval.dense") as span:
            return self._dense(vector, before, exclude, timings, span)

    def _dense(
        self,
        vector: list[float],
        before: date | None,
        exclude: str | None,
        timings: dict[str, float],
        span: Span,
    ) -> list[SearchHit]:
        start = time.perf_counter()
        params = {
            "vec": to_pgvector(vector),
            "before": before,
            "exclude": exclude,
            "limit": CANDIDATES,
        }
        with self.conn.transaction():
            self.conn.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)", (str(HNSW_EF_SEARCH),)
            )
            if self._supports_iterative_scan():
                self.conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
            rows = self.conn.execute(_DENSE_SQL, params).fetchall()
        # relaxed_order may return near-ties slightly out of order; sort exactly.
        rows.sort(key=lambda r: (-float(r[8]), r[0]))
        timings["dense_ms"] = _ms(start)
        span.set_attribute("retrieval.dense_ms", timings["dense_ms"])
        span.set_attribute("retrieval.candidates", len(rows))
        return [_hit(r, float(r[8]), i) for i, r in enumerate(rows, start=1)]

    def lexical(
        self, query: str, before: date | None, exclude: str | None, timings: dict[str, float]
    ) -> list[SearchHit]:
        with tracer.start_as_current_span("retrieval.lexical") as span:
            hits = self._lexical(query, before, exclude, timings)
            span.set_attribute("retrieval.lexical_ms", timings["lexical_ms"])
            span.set_attribute("retrieval.candidates", len(hits))
            return hits

    def _lexical(
        self, query: str, before: date | None, exclude: str | None, timings: dict[str, float]
    ) -> list[SearchHit]:
        start = time.perf_counter()
        if self._n_docs is None:
            self._n_docs = history_passage_count(self.conn)
        tsq = build_or_tsquery(salient_terms(self.conn, query, n_docs=self._n_docs))
        rows = []
        if tsq is not None:
            params = {
                "tsq": tsq,
                "norm": TS_RANK_NORMALIZATION,
                "before": before,
                "exclude": exclude,
                "limit": CANDIDATES,
            }
            with self.conn.transaction():
                self.conn.execute(
                    "SELECT set_config('max_parallel_workers_per_gather', %s, true),"
                    " set_config('min_parallel_table_scan_size', %s, true)",
                    (str(LEXICAL_PARALLEL_WORKERS), LEXICAL_MIN_PARALLEL_SCAN),
                )
                rows = self.conn.execute(_LEXICAL_SQL, params).fetchall()
        timings["lexical_ms"] = _ms(start)
        return [_hit(r, float(r[8]), i) for i, r in enumerate(rows, start=1)]

    def ranked(
        self,
        query: str,
        *,
        before_date: date | None,
        exclude_event_id: str | None,
        mode: Mode,
        timings: dict[str, float],
        notes: list[str] | None = None,
    ) -> list[SearchHit]:
        """The full ranked candidate list (up to CANDIDATES passages) for ``mode``.

        With ``rerank_fallback`` a reranker failure returns the RRF list and appends the
        reason to ``notes``.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not query.strip():
            return []
        if mode == "dense":
            return self.dense(query, before_date, exclude_event_id, timings)
        if mode == "lexical":
            return self.lexical(query, before_date, exclude_event_id, timings)

        dense = self.dense(query, before_date, exclude_event_id, timings)
        lexical = self.lexical(query, before_date, exclude_event_id, timings)
        with tracer.start_as_current_span("retrieval.fuse") as span:
            start = time.perf_counter()
            by_id = {h.passage_id: h for h in [*lexical, *dense]}
            fused = rrf_fuse([[h.passage_id for h in dense], [h.passage_id for h in lexical]])
            hits = [
                _replace(by_id[pid], score=score, rank=i) for i, (pid, score) in enumerate(fused, 1)
            ]
            timings["fuse_ms"] = _ms(start)
            span.set_attribute("retrieval.fuse_ms", timings["fuse_ms"])
        if mode == "rrf":
            return hits

        with tracer.start_as_current_span("retrieval.rerank") as span:
            start = time.perf_counter()
            try:
                scores = self.reranker.score(query, [h.content for h in hits])
                if len(scores) != len(hits):
                    raise ValueError(f"reranker returned {len(scores)} scores for {len(hits)}")
            except Exception as err:  # any reranker failure degrades to the RRF ranking
                if not self.rerank_fallback:
                    raise
                timings["rerank_ms"] = _ms(start)
                reason = f"rerank failed ({type(err).__name__}: {err}); returned the rrf ranking"
                log.warning(reason)
                span.record_exception(err)
                span.set_status(Status(StatusCode.ERROR, reason))
                span.set_attribute("retrieval.fallback", "rrf")
                if notes is not None:
                    notes.append(reason)
                return hits
            order = sorted(range(len(hits)), key=lambda i: (-scores[i], hits[i].rank))
            timings["rerank_ms"] = _ms(start)
            span.set_attribute("retrieval.rerank_ms", timings["rerank_ms"])
        return [_replace(hits[i], score=scores[i], rank=r) for r, i in enumerate(order, 1)]

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        before_date: date | None,
        exclude_event_id: str | None = None,
        mode: Mode = "rrf_rerank",
        history_only: bool = True,
    ) -> SearchResult:
        """Top ``top_k`` passages for ``query`` from events strictly before ``before_date``.

        ``before_date`` is required on purpose. With ``history_only`` (the default) it is
        capped at HELD_OUT_FROM; evaluations and the agent never pass ``history_only=False``.
        """
        _check_top_k(top_k)
        timings: dict[str, float] = {}
        notes: list[str] = []
        with tracer.start_as_current_span("retrieval.search") as span:
            start = time.perf_counter()
            hits = self.ranked(
                query,
                before_date=effective_before_date(before_date, history_only),
                exclude_event_id=exclude_event_id,
                mode=mode,
                timings=timings,
                notes=notes,
            )
            timings["total_ms"] = _ms(start)
            _annotate(span, mode, top_k, timings, notes)
        return SearchResult(
            hits=hits[:top_k], timings_ms=timings, fallback=notes[0] if notes else None
        )

    def search_events(
        self,
        query: str,
        top_k: int = 10,
        *,
        before_date: date | None,
        exclude_event_id: str | None = None,
        mode: Mode = "rrf_rerank",
        history_only: bool = True,
    ) -> EventSearchResult:
        """Top ``top_k`` distinct events, ranked by their best passage in the full list.

        ``before_date`` is capped at HELD_OUT_FROM unless ``history_only=False``.
        """
        _check_top_k(top_k)
        timings: dict[str, float] = {}
        notes: list[str] = []
        with tracer.start_as_current_span("retrieval.search_events") as span:
            start = time.perf_counter()
            hits = self.ranked(
                query,
                before_date=effective_before_date(before_date, history_only),
                exclude_event_id=exclude_event_id,
                mode=mode,
                timings=timings,
                notes=notes,
            )
            grouped: dict[str, list[SearchHit]] = {}
            for h in hits:
                grouped.setdefault(h.event_id, []).append(h)
            events = [
                EventHit(event_id=ev, score=hs[0].score, rank=r, hits=hs)
                for r, (ev, hs) in enumerate(grouped.items(), start=1)
            ]
            timings["total_ms"] = _ms(start)
            _annotate(span, mode, top_k, timings, notes)
        return EventSearchResult(
            events=events[:top_k], timings_ms=timings, fallback=notes[0] if notes else None
        )


def _annotate(
    span: Span, mode: Mode, top_k: int, timings: dict[str, float], notes: list[str]
) -> None:
    span.set_attribute("retrieval.mode", mode)
    span.set_attribute("retrieval.top_k", top_k)
    for key, ms in timings.items():
        span.set_attribute(f"retrieval.{key}", ms)
    if notes:
        span.set_attribute("retrieval.fallback", "rrf")


def effective_before_date(before_date: date | None, history_only: bool) -> date | None:
    """The cutoff a search applies: capped at HELD_OUT_FROM (via corpus_cutoff) by default."""
    if not history_only:
        return before_date
    return HELD_OUT_FROM if before_date is None else corpus_cutoff(before_date)


def _replace(hit: SearchHit, *, score: float, rank: int) -> SearchHit:
    return SearchHit(hit.passage_id, hit.event_id, score, rank, hit.content, hit.metadata)


def _check_top_k(top_k: int) -> None:
    if not 1 <= top_k <= CANDIDATES:
        raise ValueError(f"top_k must be between 1 and {CANDIDATES}, got {top_k}")


def search(
    query: str,
    top_k: int = 10,
    *,
    before_date: date | None,
    exclude_event_id: str | None = None,
    mode: Mode = "rrf_rerank",
    history_only: bool = True,
    conn: Connection | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> SearchResult:
    """Module-level ``Retriever.search``; opens a connection from DATABASE_URL if needed."""
    if conn is None:
        with connect() as own:
            return search(
                query,
                top_k,
                before_date=before_date,
                exclude_event_id=exclude_event_id,
                mode=mode,
                history_only=history_only,
                conn=own,
                embedder=embedder,
                reranker=reranker,
            )
    return Retriever(conn, embedder, reranker).search(
        query,
        top_k,
        before_date=before_date,
        exclude_event_id=exclude_event_id,
        mode=mode,
        history_only=history_only,
    )


def search_events(
    query: str,
    top_k: int = 10,
    *,
    before_date: date | None,
    exclude_event_id: str | None = None,
    mode: Mode = "rrf_rerank",
    history_only: bool = True,
    conn: Connection | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> EventSearchResult:
    """Module-level ``Retriever.search_events``."""
    if conn is None:
        with connect() as own:
            return search_events(
                query,
                top_k,
                before_date=before_date,
                exclude_event_id=exclude_event_id,
                mode=mode,
                history_only=history_only,
                conn=own,
                embedder=embedder,
                reranker=reranker,
            )
    return Retriever(conn, embedder, reranker).search_events(
        query,
        top_k,
        before_date=before_date,
        exclude_event_id=exclude_event_id,
        mode=mode,
        history_only=history_only,
    )
