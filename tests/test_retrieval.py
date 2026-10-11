"""Retrieval: RRF, lexical query building, embedding cache/resume and leakage filters."""

from collections.abc import Iterator
from datetime import date
from typing import Any

import psycopg
import pytest
from conftest import FIXTURES, reset
from fakes import FakeEmbedder, FakeReranker

from causetrace.db import Connection
from causetrace.embeddings import embed_passages
from causetrace.index import build_index
from causetrace.ingest import ingest
from causetrace.packets import get_incident_packet
from causetrace.retrieval import (
    MODES,
    Mode,
    Retriever,
    build_or_tsquery,
    effective_before_date,
    rrf_fuse,
    salient_terms,
    search,
)
from causetrace.split import HELD_OUT_FROM, load_split

# --- RRF on hand-made lists -------------------------------------------------------


def test_rrf_scores_are_sum_of_reciprocal_ranks() -> None:
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)


def test_rrf_order_and_ties() -> None:
    # b is in both lists and wins; a and x tie on score and best rank, so id decides.
    assert [i for i, _ in rrf_fuse([["a", "b"], ["x", "b"]])] == ["b", "a", "x"]
    # c (rank 1 + rank 3) beats d (rank 2 + rank 2): 1/61+1/63 > 2/62.
    assert [i for i, _ in rrf_fuse([["c", "d"], ["e", "d", "c"]])][:2] == ["c", "d"]


def test_rrf_limit_and_empty() -> None:
    long = [f"p{i}" for i in range(80)]
    assert len(rrf_fuse([long, long[::-1]], limit=50)) == 50
    assert rrf_fuse([[], []]) == []
    assert [(i, round(s, 9)) for i, s in rrf_fuse([["a"], []])] == [("a", round(1 / 61, 9))]


def test_build_or_tsquery() -> None:
    assert build_or_tsquery(["engin", "power"]) == "'engin' | 'power'"
    assert build_or_tsquery(["o'hare"]) == "'o''hare'"
    assert build_or_tsquery([]) is None


# --- Fixture index with fake models ----------------------------------------------


@pytest.fixture(scope="module")
def indexed(db_url: str) -> Iterator[tuple[Connection, FakeEmbedder]]:
    with psycopg.connect(db_url, autocommit=True) as conn:
        reset(conn)
        ingest(conn, FIXTURES)
        embedder = FakeEmbedder()
        _, _, stats = build_index(conn, embedder, batch_size=128)
        assert stats is not None and stats.remaining == 0
        yield conn, embedder


def test_salient_terms_are_capped_and_in_corpus(
    indexed: tuple[Connection, FakeEmbedder],
) -> None:
    conn, _ = indexed
    packet = get_incident_packet("20210312102750", conn)
    narrative = " ".join(a.narr_accp or "" for a in packet.aircraft)
    terms = salient_terms(conn, narrative)
    assert 0 < len(terms) <= 30
    known = {r[0] for r in conn.execute("SELECT lexeme FROM lexeme_stats").fetchall()}
    assert set(terms) <= known
    assert salient_terms(conn, "the of and") == []  # stopwords only


def test_whole_narrative_lexical_query_matches(indexed: tuple[Connection, FakeEmbedder]) -> None:
    conn, embedder = indexed
    text = conn.execute(
        "SELECT content FROM passages WHERE passage_id = '20080219X00204:factual:0'"
    ).fetchone()
    assert text is not None
    result = Retriever(conn, embedder).search(text[0], before_date=None, mode="lexical")
    assert len(result.hits) == 10
    assert result.hits[0].event_id == "20080219X00204"


def _queries(conn: Connection) -> list[tuple[str, str, date]]:
    """(ev_id, packet narrative, before_date) for a spread of held-out and history events."""
    split = load_split(conn)
    targets = list(split.held_out[::10]) + list(split.historical[50::25])
    dates: dict[str, date] = dict(conn.execute("SELECT ev_id, ev_date FROM events").fetchall())
    out = []
    for ev_id in targets:
        packet = get_incident_packet(ev_id, conn)
        narrative = "\n".join(a.narr_accp for a in packet.aircraft if a.narr_accp)
        query = narrative or f"{packet.aircraft[0].make} {packet.aircraft[0].model} accident"
        before = min(dates[ev_id], HELD_OUT_FROM)
        out.append((ev_id, query, before))
    return out


@pytest.mark.parametrize("mode", MODES)
def test_excluded_event_and_dates_are_enforced(
    indexed: tuple[Connection, FakeEmbedder], mode: Mode
) -> None:
    conn, embedder = indexed
    retriever = Retriever(conn, embedder, FakeReranker())
    dates: dict[str, date] = dict(conn.execute("SELECT ev_id, ev_date FROM events").fetchall())
    queries = _queries(conn)
    assert len(queries) >= 15
    for ev_id, query, before in queries:
        result = retriever.search(query, 50, before_date=before, exclude_event_id=ev_id, mode=mode)
        events = retriever.search_events(
            query, 50, before_date=before, exclude_event_id=ev_id, mode=mode
        )
        assert result.hits, (mode, ev_id)
        for hit in result.hits:
            assert hit.event_id != ev_id
            assert dates[hit.event_id] < before  # checked against events, not hit metadata
            assert hit.metadata["event_date"] < before
        assert [h.rank for h in result.hits] == list(range(1, len(result.hits) + 1))
        ids = [e.event_id for e in events.events]
        assert len(ids) == len(set(ids))
        assert ev_id not in ids and all(dates[e] < before for e in ids)
        assert "total_ms" in result.timings_ms


def test_exclusion_removes_a_self_match(indexed: tuple[Connection, FakeEmbedder]) -> None:
    conn, embedder = indexed
    text = conn.execute(
        "SELECT content FROM passages WHERE passage_id = '20080219X00204:probable_cause:0'"
    ).fetchone()
    assert text is not None
    retriever = Retriever(conn, embedder, FakeReranker())
    for mode in MODES:
        top = retriever.search(text[0], 5, before_date=None, mode=mode)
        assert top.hits[0].event_id == "20080219X00204", mode
        excluded = retriever.search(
            text[0],
            50,
            before_date=None,
            exclude_event_id="20080219X00204",
            mode=mode,
        )
        assert all(h.event_id != "20080219X00204" for h in excluded.hits)
        early = retriever.search(text[0], 50, before_date=date(2008, 1, 11), mode=mode)
        assert all(h.metadata["event_date"] < date(2008, 1, 11) for h in early.hits)


def test_stage_timings_and_modes(indexed: tuple[Connection, FakeEmbedder]) -> None:
    conn, embedder = indexed
    reranker = FakeReranker()
    result = search(
        "loss of engine power during takeoff",
        before_date=HELD_OUT_FROM,
        mode="rrf_rerank",
        conn=conn,
        embedder=embedder,
        reranker=reranker,
    )
    assert len(result.hits) == 10
    assert {"embed_ms", "dense_ms", "lexical_ms", "fuse_ms", "rerank_ms", "total_ms"} <= set(
        result.timings_ms
    )
    assert reranker.calls == 1
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
    lexical = search("engine", before_date=None, mode="lexical", conn=conn)
    assert "embed_ms" not in lexical.timings_ms  # no model needed
    with pytest.raises(ValueError):
        search("x", 0, before_date=None, mode="dense", conn=conn, embedder=embedder)
    with pytest.raises(ValueError):
        search("x", before_date=None, mode="bogus", conn=conn)  # type: ignore[arg-type]


def test_hits_carry_metadata(indexed: tuple[Connection, FakeEmbedder]) -> None:
    conn, embedder = indexed
    hit = Retriever(conn, embedder).search("helicopter", 1, before_date=None, mode="dense").hits[0]
    assert set(hit.metadata) == {
        "event_date",
        "section_type",
        "chunk_index",
        "aircraft_category",
        "phase_of_flight",
    }
    assert hit.passage_id.startswith(hit.event_id + ":")


# --- Embedding cache and resume --------------------------------------------------


class _Interrupted(Exception):
    pass


class _FailingEmbedder(FakeEmbedder):
    def __init__(self, fail_after: int) -> None:
        super().__init__()
        self.fail_after = fail_after

    def embed(self, texts: Any, *, is_query: bool = False) -> list[list[float]]:
        if self.calls >= self.fail_after:
            raise _Interrupted
        return super().embed(texts, is_query=is_query)


def test_embedding_resumes_and_uses_cache(db: Connection) -> None:
    ingest(db, FIXTURES)
    build_index(db, None)
    total = db.execute("SELECT count(DISTINCT content_sha256) FROM passages").fetchone()
    assert total is not None

    with pytest.raises(_Interrupted):
        embed_passages(db, _FailingEmbedder(fail_after=2), batch_size=50)
    done = db.execute("SELECT count(*) FROM embedding_cache").fetchone()
    assert done == (100,)  # two committed batches survive the interruption

    resumed = FakeEmbedder()
    stats = embed_passages(db, resumed, batch_size=50)
    assert stats.remaining == 0
    assert resumed.texts_embedded == total[0] - 100

    # Wipe passage embeddings: everything comes back from the cache, nothing is embedded.
    db.execute("UPDATE passages SET embedding = NULL, embedding_model = NULL")
    again = FakeEmbedder()
    stats = embed_passages(db, again, batch_size=50)
    assert again.texts_embedded == 0 and stats.remaining == 0 and stats.from_cache > 0

    # A different model revision is not served from the old cache.
    other = FakeEmbedder()
    other.revision = "v2"
    stats = embed_passages(db, other, batch_size=500, max_batches=1)
    assert other.texts_embedded == min(500, total[0])


def test_effective_before_date() -> None:
    assert effective_before_date(date(2025, 6, 1), True) == HELD_OUT_FROM
    assert effective_before_date(date(2015, 6, 1), True) == date(2015, 6, 1)
    assert effective_before_date(None, True) == HELD_OUT_FROM
    assert effective_before_date(date(2025, 6, 1), False) == date(2025, 6, 1)
    assert effective_before_date(None, False) is None


@pytest.mark.parametrize("mode", MODES)
def test_history_only_caps_a_held_out_before_date(
    indexed: tuple[Connection, FakeEmbedder], mode: Mode
) -> None:
    """A held-out query whose before_date is its own date still sees history only."""
    conn, embedder = indexed
    row = conn.execute(
        "SELECT ev_id, ev_date FROM events WHERE ev_date >= %s"
        " ORDER BY ev_date DESC, ev_id LIMIT 1",
        (HELD_OUT_FROM,),
    ).fetchone()
    assert row is not None
    ev_id, ev_date = row
    assert ev_date > HELD_OUT_FROM
    packet = get_incident_packet(ev_id, conn)
    query = "\n".join(a.narr_accp for a in packet.aircraft if a.narr_accp)
    retriever = Retriever(conn, embedder, FakeReranker())

    capped = retriever.search(query, 50, before_date=ev_date, exclude_event_id=ev_id, mode=mode)
    assert capped.hits
    assert all(h.metadata["event_date"] < HELD_OUT_FROM for h in capped.hits)
    events = retriever.search_events(
        query, 50, before_date=ev_date, exclude_event_id=ev_id, mode=mode
    )
    dates: dict[str, date] = dict(conn.execute("SELECT ev_id, ev_date FROM events").fetchall())
    assert events.events and all(dates[e.event_id] < HELD_OUT_FROM for e in events.events)

    # Only an explicit history_only=False lifts the cap (held-out passages then appear).
    lifted = retriever.search(
        query, 50, before_date=ev_date, exclude_event_id=ev_id, mode=mode, history_only=False
    )
    assert any(h.metadata["event_date"] >= HELD_OUT_FROM for h in lifted.hits)
    assert all(h.metadata["event_date"] < ev_date for h in lifted.hits)
