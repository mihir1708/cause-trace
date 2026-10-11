"""Known-item question tooling: sampling, drafting (fake provider) and the review loop."""

from collections import Counter
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
import pytest
from conftest import FIXTURES, reset

from causetrace.db import Connection
from causetrace.evaluation.questions import (
    BOILERPLATE,
    MIN_WORDS,
    draft_all,
    kept_questions,
    read_jsonl,
    review_loop,
    sample_passages,
)
from causetrace.index import build_index
from causetrace.ingest import ingest
from causetrace.llm import BudgetExceededError, SpendTracker
from causetrace.llm.fake import FakeProvider
from causetrace.split import HELD_OUT_FROM


@pytest.fixture(scope="module")
def passages_db(db_url: str) -> Iterator[Connection]:
    with psycopg.connect(db_url, autocommit=True) as conn:
        reset(conn)
        ingest(conn, FIXTURES)
        build_index(conn, None)
        yield conn


def test_sample_is_stratified_deterministic_and_history_only(passages_db: Connection) -> None:
    sample = sample_passages(passages_db, 220, seed=42)
    assert len(sample) == 220
    assert len({p.passage_id for p in sample}) == 220
    assert Counter(p.section_type for p in sample) == {
        "factual": 74,
        "analysis": 73,
        "probable_cause": 73,
    }
    dates: dict[str, date] = dict(
        passages_db.execute("SELECT passage_id, event_date FROM passages").fetchall()
    )
    assert all(dates[p.passage_id] < HELD_OUT_FROM for p in sample)
    assert sample == sample_passages(passages_db, 220, seed=42)
    assert sample != sample_passages(passages_db, 220, seed=7)

    # Within a section, years follow the passage counts (largest remainder: within 1).
    rows = passages_db.execute(
        "SELECT extract(year FROM event_date)::int, count(*) FROM passages"
        " WHERE section_type = 'analysis' AND event_date < %s GROUP BY 1",
        (HELD_OUT_FROM,),
    ).fetchall()
    total = sum(n for _, n in rows)
    got = Counter(p.year for p in sample if p.section_type == "analysis")
    for year, n in rows:
        assert abs(got[year] - 73 * n / total) < 1


def test_sample_skips_passages_that_cannot_identify_one_event(passages_db: Connection) -> None:
    rows = passages_db.execute(
        "SELECT passage_id, event_id, content FROM passages WHERE event_date < %s",
        (HELD_OUT_FROM,),
    ).fetchall()
    events_by_text: dict[str, set[str]] = {}
    for _, ev, content in rows:
        events_by_text.setdefault(" ".join(content.lower().split()), set()).add(ev)
    short = {pid for pid, _, content in rows if len(content.split()) < MIN_WORDS}
    assert short  # the fixture has short probable-cause passages, so the filter is exercised

    sample = sample_passages(passages_db, 220, seed=42)
    for p in sample:
        assert len(p.content.split()) >= MIN_WORDS
        assert events_by_text[" ".join(p.content.lower().split())] == {p.event_id}
        assert not any(marker in p.content.lower() for marker in BOILERPLATE)
    assert not short & {p.passage_id for p in sample}


def _responder(system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"question": f"What happened in a {prompt.split()[1]} passage?"}


def test_draft_all_is_resumable_and_respects_budget(
    passages_db: Connection, tmp_path: Path
) -> None:
    sample = sample_passages(passages_db, 12, seed=1)
    out = tmp_path / "drafts.jsonl"
    tracker = SpendTracker(max_usd=0.05)
    provider = FakeProvider(_responder, tracker=tracker, usd_per_call=0.01)
    with pytest.raises(BudgetExceededError):
        draft_all(provider, sample, out, log=lambda _: None)
    assert len(read_jsonl(out)) == 5  # five calls fit the budget, all saved

    provider = FakeProvider(_responder)
    written, failed = draft_all(provider, sample, out, log=lambda _: None)
    assert (written, failed) == (7, 0)
    assert len(provider.prompts) == 7  # already drafted passages are not re-asked
    records = read_jsonl(out)
    assert [r["passage_id"] for r in records] == [p.passage_id for p in sample]
    assert set(records[0]) == {
        "passage_id",
        "event_id",
        "section_type",
        "year",
        "question",
        "usage",
    }
    assert records[0]["event_id"] == sample[0].event_id


def test_empty_question_is_skipped(passages_db: Connection, tmp_path: Path) -> None:
    sample = sample_passages(passages_db, 3, seed=1)
    logged: list[str] = []
    provider = FakeProvider(lambda s, p, sc: {"question": "  "})
    assert draft_all(provider, sample, tmp_path / "d.jsonl", log=logged.append) == (0, 3)
    assert len(logged) == 3


def _drafts(n: int) -> list[dict[str, Any]]:
    return [
        {"passage_id": f"E{i}:analysis:0", "event_id": f"E{i}", "section_type": "analysis",
         "question": f"draft {i}?"}
        for i in range(n)
    ]  # fmt: skip


def test_review_loop_records_and_resumes(tmp_path: Path) -> None:
    drafts = _drafts(4)
    out = tmp_path / "reviewed.jsonl"
    answers = iter(["k", "x", "e", "", "edited question?", "d", "q"])
    shown: list[str] = []
    recorded = review_loop(
        drafts, out, lambda pid: f"passage {pid}", ask=lambda _: next(answers), show=shown.append
    )
    assert recorded == 3
    assert any("passage E0:analysis:0" in s for s in shown)
    assert any("QUESTION: draft 0?" in s for s in shown)
    records = read_jsonl(out)
    assert [r["decision"] for r in records] == ["keep", "edit", "drop"]
    assert [r["question"] for r in records] == ["draft 0?", "edited question?", None]
    assert records[1]["draft_question"] == "draft 1?"

    # Resume: only the fourth draft is shown; EOF stops cleanly.
    answers = iter(["k"])
    recorded = review_loop(
        drafts, out, lambda pid: None, ask=lambda _: next(answers), show=lambda _: None
    )
    assert recorded == 1
    assert review_loop(drafts, out, lambda pid: None, ask=lambda _: "k", show=lambda _: None) == 0

    def eof(_: str) -> str:
        raise EOFError

    assert review_loop(_drafts(5), out, lambda pid: None, ask=eof, show=lambda _: None) == 0

    kept = kept_questions(read_jsonl(out))
    assert [k["question"] for k in kept] == ["draft 0?", "edited question?", "draft 3?"]
