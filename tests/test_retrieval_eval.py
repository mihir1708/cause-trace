"""Retrieval evaluation runner on the fixture with fake models (no downloads)."""

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import psycopg
import pytest
from conftest import FIXTURES, reset
from fakes import FakeEmbedder, FakeReranker

from causetrace.db import Connection
from causetrace.evaluation import retrieval_eval as ev
from causetrace.index import build_index
from causetrace.ingest import ingest
from causetrace.retrieval import MODES, Retriever
from causetrace.split import HELD_OUT_FROM, corpus_cutoff


@pytest.fixture(scope="module")
def indexed(db_url: str) -> Iterator[Connection]:
    with psycopg.connect(db_url, autocommit=True) as conn:
        reset(conn)
        ingest(conn, FIXTURES)
        build_index(conn, FakeEmbedder(), batch_size=256)
        yield conn


def _retriever(conn: Connection) -> Retriever:
    return Retriever(conn, FakeEmbedder(), FakeReranker())


def test_corpus_cutoff() -> None:
    assert corpus_cutoff(date(2025, 3, 1)) == HELD_OUT_FROM
    assert corpus_cutoff(date(2015, 3, 1)) == date(2015, 3, 1)


def test_scored_held_out_events(indexed: Connection) -> None:
    events = ev.scored_held_out_events(indexed)
    assert len(events) == 100  # every fixture held-out event has labels and narr_accp
    assert all(d >= HELD_OUT_FROM for _, d in events)
    assert events == sorted(events, key=lambda e: (e[1], e[0]))
    assert ev.select_events(events, 10, 3) == ev.select_events(events, 10, 3)
    assert len(ev.select_events(events, 10, 3)) == 10
    assert ev.select_events(events, None, 3) == events


def test_evaluate_incidents_all_modes(indexed: Connection) -> None:
    events = ev.select_events(ev.scored_held_out_events(indexed), 12, 0)
    result = ev.evaluate_incidents(indexed, _retriever(indexed), events, MODES)
    assert result["n_queries"] == 12
    assert set(result["modes"]) == set(MODES)
    for r in result["modes"].values():
        for key in ("label_recall_at_10", "ndcg_at_10", "mrr", "hit_at_10"):
            assert 0.0 <= r[key] <= 1.0
        assert r["p50_ms"] <= r["p95_ms"]


def test_evaluate_questions_known_item(indexed: Connection) -> None:
    rows = indexed.execute(
        "SELECT event_id, content FROM passages WHERE section_type = 'probable_cause'"
        " AND event_date < %s ORDER BY passage_id LIMIT 5",
        (HELD_OUT_FROM,),
    ).fetchall()
    questions = [{"event_id": e, "question": c} for e, c in rows]
    questions.append({"event_id": "29990101X00000", "question": "unknown event"})
    known = {
        r[0]
        for r in indexed.execute("SELECT ev_id FROM events WHERE ev_date < %s", (HELD_OUT_FROM,))
    }
    result = ev.evaluate_questions(_retriever(indexed), questions, MODES, known)
    assert result["n_queries"] == 5 and result["n_skipped_unknown_event"] == 1
    # The question is the source passage itself, so lexical retrieval finds it first.
    assert result["modes"]["lexical"]["recall_at_10"] == 1.0
    assert result["modes"]["lexical"]["mrr"] == 1.0


def test_check_gate() -> None:
    base = {"label_recall_at_10": 0.50, "query_set_sha256": "x", "n_queries": 100}
    ok, _ = ev.check_gate({**base, "label_recall_at_10": 0.48}, base)
    assert ok  # a drop of exactly 0.02 passes
    ok, msg = ev.check_gate({**base, "label_recall_at_10": 0.479}, base)
    assert not ok and "drop -0.0210" not in msg and "0.4790" in msg
    ok, _ = ev.check_gate({**base, "label_recall_at_10": 0.60}, base)
    assert ok
    ok, msg = ev.check_gate({**base, "query_set_sha256": "y"}, base)
    assert not ok and "update-baseline" in msg
    ok, msg = ev.check_gate(base, {**base, "mode": "rrf_rerank"})
    assert not ok and "update-baseline" in msg


def test_main_writes_results_and_gate(indexed: Connection, db_url: str, tmp_path: Path) -> None:
    rows = indexed.execute(
        "SELECT passage_id, event_id, content FROM passages WHERE section_type = 'analysis'"
        " AND event_date < %s ORDER BY passage_id LIMIT 3",
        (HELD_OUT_FROM,),
    ).fetchall()
    questions = tmp_path / "reviewed.jsonl"
    with questions.open("w") as f:
        for pid, ev_id, content in rows:
            f.write(json.dumps({"passage_id": pid, "event_id": ev_id, "decision": "keep",
                                "question": content[:300]}) + "\n")  # fmt: skip
        f.write(
            json.dumps({"passage_id": "x", "event_id": "y", "decision": "drop", "question": None})
            + "\n"
        )

    common = ["--database-url", db_url, "--limit", "8", "--seed", "5"]
    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    code = ev.main(
        [
            *common,
            "--questions",
            str(questions),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        retriever_factory=_retriever,
    )
    assert code == 0
    report = json.loads(out_json.read_text())
    assert report["questions"]["n_queries"] == 3
    assert report["incidents"]["n_queries"] == 8
    assert set(report["incidents"]["modes"]) == set(MODES)
    assert (
        report["snapshot_sha256"]
        == json.loads(Path("data_manifest.json").read_text())["files"]["avall.zip"]["sha256"]
    )
    assert report["models"]["embedding"].startswith("BAAI/bge-small-en-v1.5@5c38ec7")
    md = out_md.read_text()
    assert "| Mode | Recall@10 | MRR | p95 ms |" in md
    assert "| Mode | Label Recall@10 | nDCG@10 | MRR | Hit@10 | p50 ms | p95 ms |" in md
    assert md.count("| rrf_rerank |") == 2

    baseline = tmp_path / "baseline.json"
    assert ev.main([*common, "--gate", str(baseline)], retriever_factory=_retriever) == 1  # missing
    assert ev.main([*common, "--write-baseline", str(baseline)], retriever_factory=_retriever) == 0
    assert ev.main([*common, "--gate", str(baseline)], retriever_factory=_retriever) == 0
    data = json.loads(baseline.read_text())
    assert data["mode"] == "rrf" and data["n_queries"] == 8
    data["label_recall_at_10"] += 0.05
    baseline.write_text(json.dumps(data))
    assert ev.main([*common, "--gate", str(baseline)], retriever_factory=_retriever) == 1
