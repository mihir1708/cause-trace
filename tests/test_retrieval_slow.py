"""Real models on the fixture. Needs the `ml` group and models already in the Hugging Face
cache (`make models`); never downloads, and skips otherwise."""

from datetime import date
from typing import Any

import pytest
from conftest import FIXTURES

from causetrace.config import EMBEDDING_DIM
from causetrace.db import Connection
from causetrace.index import build_index
from causetrace.ingest import ingest
from causetrace.retrieval import MODES, Retriever
from causetrace.split import HELD_OUT_FROM, load_split

pytestmark = pytest.mark.slow


def _models() -> tuple[Any, Any]:
    pytest.importorskip("sentence_transformers", reason="needs the `ml` dependency group")
    from causetrace.embeddings import SentenceTransformerEmbedder
    from causetrace.rerank import CrossEncoderReranker

    try:
        return (
            SentenceTransformerEmbedder(local_files_only=True),
            CrossEncoderReranker(local_files_only=True),
        )
    except Exception as err:  # not cached / cannot be loaded offline
        pytest.skip(f"retrieval models not available offline ({type(err).__name__}: {err})")


def test_real_models_on_fixture(db: Connection) -> None:
    embedder, reranker = _models()
    vec = embedder.embed(["engine failure on takeoff"], is_query=True)[0]
    assert len(vec) == EMBEDDING_DIM
    assert sum(x * x for x in vec) == pytest.approx(1.0, abs=1e-4)

    ingest(db, FIXTURES)
    _, _, stats = build_index(db, embedder, batch_size=32)
    assert stats is not None and stats.remaining == 0

    retriever = Retriever(db, embedder, reranker)
    own = db.execute(
        "SELECT content FROM passages WHERE passage_id = '20080219X00204:probable_cause:0'"
    ).fetchone()
    assert own is not None
    assert retriever.search(own[0], 1, before_date=None, mode="dense").hits[0].event_id == (
        "20080219X00204"
    )

    dates: dict[str, date] = dict(db.execute("SELECT ev_id, ev_date FROM events").fetchall())
    for ev_id in load_split(db).held_out[::20]:
        query = db.execute(
            "SELECT string_agg(narr_accp_safe, ' ') FROM narratives WHERE ev_id = %s", (ev_id,)
        ).fetchone()
        assert query is not None and query[0]
        for mode in MODES:
            result = retriever.search(
                query[0], before_date=HELD_OUT_FROM, exclude_event_id=ev_id, mode=mode
            )
            assert len(result.hits) == 10, mode
            assert all(
                h.event_id != ev_id and dates[h.event_id] < HELD_OUT_FROM for h in result.hits
            )
