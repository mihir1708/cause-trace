"""Causal-system comparison: metrics, free systems, LLM systems (FakeProvider), resume."""

import json
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import psycopg
import pytest
from conftest import FIXTURES, reset
from fakes import FakeEmbedder, FakeReranker

from causetrace.agent import Agent
from causetrace.agent.taxonomy import load_taxonomy
from causetrace.db import Connection
from causetrace.evaluation import causal_eval as ce
from causetrace.evaluation.causal_metrics import (
    PassageInfo,
    aggregate_repeats,
    gold_outside,
    score_cases,
)
from causetrace.evaluation.causal_systems import (
    ClosedBook,
    FrequencyPrior,
    KnnVote,
    Rag,
    RagVerify,
    RandomContext,
    apply_cutoff,
    knn_neighbours,
    knn_predict,
    knn_scores,
    random_passages,
    tune_confidence,
    tune_tau,
)
from causetrace.evaluation.citation_audit import (
    audit_summary,
    grade_loop,
    read_audit,
    write_audit_sample,
)
from causetrace.gold import get_all_gold_labels
from causetrace.index import build_index
from causetrace.ingest import ingest
from causetrace.llm import LLMResponseError, ProviderUnavailableError, SpendTracker
from causetrace.llm.fake import FakeProvider
from causetrace.retrieval import Retriever
from causetrace.split import HELD_OUT_FROM, corpus_cutoff

A, B, C, D = "01-01-01", "02-02-02", "03-03-03", "04-04-04"
PASSAGE_ID = re.compile(r"^\[([^\]]+)\] ", re.M)
CODE = re.compile(r"^(\d{2}-\d{2}-\d{2})  ", re.M)

# --- Metrics on hand-made cases ---------------------------------------------------------


def test_score_cases_hand_computed() -> None:
    gold = {"e1": frozenset({A, B}), "e2": frozenset({C}), "e3": frozenset({A, D})}
    results = [
        {"ev_id": "e1", "labels": [A, C], "abstained": False},  # tp 1, fp 1, fn 1
        {"ev_id": "e2", "labels": [], "abstained": True},  # fn 1
        {"ev_id": "e3", "labels": [A, D], "abstained": False},  # tp 2
    ]
    cut = {ev: HELD_OUT_FROM for ev in gold}
    m = score_cases(results, gold, history_gold={}, passages={}, cutoffs=cut)
    assert (m["recovered"], m["gold_findings"], m["predicted"]) == (3, 5, 4)
    assert m["micro_precision"] == pytest.approx(3 / 4)
    assert m["micro_recall"] == pytest.approx(3 / 5)
    assert m["micro_f1"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 2))
    # Answered cases only: e1 and e3 -> tp 3, fp 1, fn 1.
    assert m["micro_f1_answered"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 1))
    # Macro: e1 p=r=f=0.5; e2 all 0; e3 all 1.
    assert m["macro_precision"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx(0.5)
    assert m["abstention_rate"] == pytest.approx(1 / 3)
    assert m["citations"] is None and m["citation_validity"] is None
    assert m["verifier_support_rate"] is None


def test_citation_validity_support_and_verifier() -> None:
    gold = {"t": frozenset({A})}
    history = {"h1": frozenset({A}), "h2": frozenset({B}), "late": frozenset({A})}
    passages = {
        "h1:analysis:0": PassageInfo("h1", date(2020, 1, 1)),
        "h2:analysis:0": PassageInfo("h2", date(2021, 1, 1)),
        "late:analysis:0": PassageInfo("late", date(2024, 3, 1)),
    }
    res = {
        "ev_id": "t",
        "labels": [A],
        "abstained": False,
        "cites": True,
        "context_passages": ["h1:analysis:0", "h2:analysis:0", "late:analysis:0"],
        "citations": [
            {"label": A, "passage_id": "h1:analysis:0"},  # valid, supported
            {"label": A, "passage_id": "h2:analysis:0"},  # valid, h2 lacks A
            {"label": A, "passage_id": "late:analysis:0"},  # after the cutoff: invalid
            {"label": A, "passage_id": "x:analysis:0"},  # unknown and not in context
        ],
        "verifier": {"pairs": 4, "supported": 1},
    }
    m = score_cases(
        [res], gold, history_gold=history, passages=passages, cutoffs={"t": HELD_OUT_FROM}
    )
    assert m["citations"] == 4
    assert m["citation_validity"] == pytest.approx(2 / 4)
    assert m["citation_support"] == pytest.approx(1 / 4)
    assert m["verifier_support_rate"] == pytest.approx(1 / 4)


def test_empty_prediction_counts_as_abstention() -> None:
    m = score_cases(
        [{"ev_id": "e", "labels": [], "abstained": False}],
        {"e": frozenset({A})},
        history_gold={},
        passages={},
        cutoffs={"e": HELD_OUT_FROM},
    )
    assert m["abstention_rate"] == 1.0 and m["micro_f1"] == 0.0
    assert m["micro_f1_answered"] is None


def test_aggregate_repeats_and_gold_outside() -> None:
    agg = aggregate_repeats([{"micro_f1": 0.4, "n_cases": 10}, {"micro_f1": 0.6, "n_cases": 10}])
    assert agg["repeats"] == 2
    assert agg["micro_f1"] == pytest.approx(0.5)
    assert (agg["micro_f1_min"], agg["micro_f1_max"]) == (0.4, 0.6)
    out = gold_outside({"e1": frozenset({A, D}), "e2": frozenset({A})}, [A, B])
    assert out == {"labels": 1, "cases": 1, "distinct_codes": 1}


def test_knn_votes_and_tuning() -> None:
    gold = {"n1": frozenset({A, B}), "n2": frozenset({A}), "n3": frozenset({C})}
    scores = knn_scores(["n1", "n2", "n3"], gold)
    total = 1 + 1 / 2 + 1 / 3
    assert scores[A] == pytest.approx((1 + 1 / 2) / total)
    assert scores[B] == pytest.approx(1 / total)
    assert scores[C] == pytest.approx((1 / 3) / total)
    assert knn_predict(["n1", "n2", "n3"], gold, 0.5, 5) == [A, B]
    assert knn_predict(["n1", "n2", "n3"], gold, 0.9, 5) == []
    assert knn_predict(["n1", "n2", "n3"], gold, 0.05, 1) == [A]
    tuning_gold = {**gold, "t1": frozenset({A}), "t2": frozenset({A})}
    tau, f1 = tune_tau({"t1": ["n1", "n2", "n3"], "t2": ["n2", "n3", "n1"]}, tuning_gold, 5)
    assert f1[tau] == max(f1.values())
    assert set(knn_predict(["n1", "n2", "n3"], tuning_gold, tau, 5)) == {A}


def test_frequency_prior_tunes_k_on_2023_only() -> None:
    dates = {
        "h1": date(2010, 1, 1),
        "h2": date(2011, 1, 1),
        "h3": date(2012, 1, 1),
        "t1": date(2023, 5, 1),
        "t2": date(2023, 6, 1),
    }
    gold = {
        "h1": frozenset({A, B}),
        "h2": frozenset({A, B}),
        "h3": frozenset({A}),
        "t1": frozenset({A}),
        "t2": frozenset({A}),
    }
    prior = FrequencyPrior(gold, dates, ["t1", "t2"], HELD_OUT_FROM, date(2023, 1, 1))
    assert prior.k == 1  # only A appears in the 2023 events
    assert prior.labels == [A]  # A is in all five history events
    assert prior.predict(None, "x")["labels"] == [A]  # type: ignore[arg-type]


# --- Fixture database with fake retrieval models ---------------------------------------


@pytest.fixture(scope="module")
def indexed(db_url: str) -> Iterator[Connection]:
    with psycopg.connect(db_url, autocommit=True) as conn:
        reset(conn)
        ingest(conn, FIXTURES)
        build_index(conn, FakeEmbedder(), batch_size=256)
        yield conn


def _retriever(conn: Connection) -> Retriever:
    return Retriever(conn, FakeEmbedder(), FakeReranker())


def _held_out(conn: Connection, n: int = 3) -> list[tuple[str, date]]:
    return [
        (r[0], r[1])
        for r in conn.execute(
            "SELECT ev_id, ev_date FROM events WHERE ev_date >= %s ORDER BY ev_id LIMIT %s",
            (HELD_OUT_FROM, n),
        )
    ]


class Responder:
    """Answers every task; thread-safe enough for the runner's workers."""

    def __init__(self, verdict: str = "SUPPORTED") -> None:
        self.verdict = verdict
        self.tasks: list[str] = []

    def __call__(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        task = prompt.splitlines()[0].removeprefix("# Task: ").strip()
        self.tasks.append(task)
        codes = CODE.findall(system)
        pids = PASSAGE_ID.findall(prompt)
        if task == "plan":
            return {"queries": ["engine power loss on takeoff"]}
        if task == "check_evidence":
            return {"enough": True, "missing": "", "gap_query": ""}
        if task == "closed_book":
            return {
                "findings": [
                    {"label": codes[0], "confidence": 0.8, "rationale": "r"},
                    {"label": codes[1], "confidence": 0.6, "rationale": "r"},
                ]
            }
        if task == "analyze":
            return {
                "findings": [
                    {
                        "label": codes[i],
                        "confidence": 0.9 - i / 10,
                        "citations": [pids[i]],
                        "rationale": "r",
                    }
                    for i in range(min(2, len(pids)))
                ]
            }
        if task == "verify":
            out = []
            for block in prompt.split("Finding ")[1:]:
                label = block.split(" ", 1)[0]
                for pid in re.findall(r"passage \[([^\]]+)\]", block):
                    out.append(
                        {"label": label, "passage_id": pid, "verdict": self.verdict, "reason": "x"}
                    )
            return {"verdicts": out}
        raise AssertionError(task)


def _agent_factory(provider: FakeProvider) -> Any:
    def make(conn: Connection, verify: bool) -> Agent:
        return Agent(conn, _retriever(conn), provider, verify=verify)

    return make


def test_knn_vote_uses_history_only_neighbours(indexed: Connection) -> None:
    gold = get_all_gold_labels(indexed)
    dates: dict[str, date] = dict(indexed.execute("SELECT ev_id, ev_date FROM events").fetchall())
    history = {ev: g for ev, g in gold.items() if dates[ev] < HELD_OUT_FROM}
    knn = KnnVote(history, 0.2, lambda c, ev: knn_neighbours(_retriever(c), c, ev), {})
    for ev_id, _ in _held_out(indexed):
        out = knn.predict(indexed, ev_id)
        assert ev_id not in out["neighbours"] and len(out["neighbours"]) == 10
        assert all(dates[n] < HELD_OUT_FROM for n in out["neighbours"])
        assert set(out["labels"]) <= set().union(*(history[n] for n in out["neighbours"]))


def test_closed_book(indexed: Connection) -> None:
    provider = FakeProvider(Responder(), usd_per_call=0.002)
    system = ClosedBook(provider, load_taxonomy(indexed))
    ev_id = _held_out(indexed)[0][0]
    out = system.predict(indexed, ev_id)
    assert len(out["labels"]) == 2 and out["cites"] is False
    assert out["calls"] == 1 and out["cost_usd"] == pytest.approx(0.002)
    assert provider.calls[0]["effort"] == "medium" and provider.calls[0]["cache_system"]
    assert "passage" not in provider.calls[0]["prompt"].lower().split("## incident")[0]

    def broken(*a: Any) -> dict[str, Any]:
        raise LLMResponseError("invalid JSON")

    assert (
        ClosedBook(FakeProvider(broken), load_taxonomy(indexed)).predict(indexed, ev_id)[
            "abstained"
        ]
        is True
    )


def test_random_context_is_history_only_and_seeded(indexed: Connection) -> None:
    ev_id = _held_out(indexed)[0][0]
    a = random_passages(indexed, ev_id, HELD_OUT_FROM, 42)
    assert a == random_passages(indexed, ev_id, HELD_OUT_FROM, 42)
    assert a != random_passages(indexed, ev_id, HELD_OUT_FROM, 7)
    assert len(a) == 10 and all(p["event_id"] != ev_id for p in a)
    assert all(p["event_date"] < HELD_OUT_FROM.isoformat() for p in a)

    provider = FakeProvider(Responder())
    system = RandomContext(provider, _agent_factory(provider), seed=42)
    out = system.predict(indexed, ev_id)
    assert out["context_passages"] == [p["passage_id"] for p in a]
    assert {c["passage_id"] for c in out["citations"]} <= set(out["context_passages"])
    assert out["calls"] == 1 and provider.calls[0]["effort"] == "medium"


def test_rag_skips_verify_and_rag_verify_runs_it(indexed: Connection) -> None:
    ev_id = _held_out(indexed)[0][0]
    responder = Responder()
    provider = FakeProvider(responder)
    rag = Rag(provider, _agent_factory(provider)).predict(indexed, ev_id)
    assert "verify" not in responder.tasks and "verifier" not in rag
    assert rag["calls"] == 3 and rag["labels"]

    responder.tasks.clear()
    full = RagVerify(provider, _agent_factory(provider)).predict(indexed, ev_id)
    assert "verify" in responder.tasks
    assert full["verifier"]["pairs"] >= 1
    assert full["verifier"]["supported"] == full["verifier"]["pairs"]
    for r in (rag, full):
        assert ev_id not in {p.split(":")[0] for p in r["context_passages"]}


def test_agent_switch_only_skips_verify(indexed: Connection) -> None:
    provider = FakeProvider(Responder())
    ev_id = _held_out(indexed)[0][0]
    result = Agent(indexed, _retriever(indexed), provider, verify=False).run(ev_id)
    nodes = [t["node"] for t in result["trace"]]
    assert "verify" not in nodes and nodes[-1] == "answer"
    assert result["verification"] == {}
    assert result["findings"] and all(f["citations"] for f in result["findings"])


# --- Runner: resume, stop, repeats, models, audit --------------------------------------


def _main(
    tmp: Path,
    db_url: str,
    provider: Any,
    *extra: str,
    retriever_factory: Any = _retriever,
) -> int:
    out = "dev" if "dev" in extra else "r"
    return ce.main(
        [
            "--database-url",
            db_url,
            "--cases-path",
            str(tmp / "cases.json"),
            "--dev-cases-path",
            str(tmp / "dev_cases.json"),
            "--runs-dir",
            str(tmp / "runs"),
            "--out-json",
            str(tmp / f"{out}.json"),
            "--out-md",
            str(tmp / f"{out}.md"),
            "--audit",
            str(tmp / "audit.csv"),
            *extra,
        ],
        provider_factory=provider,
        retriever_factory=retriever_factory,
    )


class CountingFactory:
    def __init__(
        self, responder: Responder, fail_after: int | None = None, usd: float = 0.001
    ) -> None:
        self.responder = responder
        self.fail_after = fail_after
        self.usd = usd
        self.calls = 0

    def __call__(self, model: str, tracker: SpendTracker) -> FakeProvider:
        outer = self

        def respond(system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
            outer.calls += 1
            if outer.fail_after is not None and outer.calls > outer.fail_after:
                raise ProviderUnavailableError("connection refused")
            return outer.responder(system, prompt, schema)

        return FakeProvider(respond, tracker=tracker, usd_per_call=self.usd, model=model)


def test_end_to_end_resume_and_report(indexed: Connection, db_url: str, tmp_path: Path) -> None:
    factory = CountingFactory(Responder())
    assert _main(tmp_path, db_url, factory, "--split", "dev", "--cases", "3") == 0
    assert _main(tmp_path, db_url, factory, "--cases", "4", "--workers", "2") == 0
    cases = json.loads((tmp_path / "cases.json").read_text())
    assert len(cases["ev_ids"]) == 4 and cases["n_scored"] == 100
    report = json.loads((tmp_path / "r.json").read_text())
    rows = {(r["system"], r["scope"]): r for r in report["rows"]}
    for system in ("closed_book", "random_context", "rag", "rag_verify"):
        assert rows[(system, "cases")]["n_cases"] == 4 and rows[(system, "cases")]["complete"]
    # The free systems run on every scored event, and are also scored on the cases.
    assert rows[("frequency_prior", "all")]["n_cases"] == 100
    assert rows[("knn_vote", "all")]["n_cases"] == 100
    assert rows[("knn_vote", "cases")]["n_cases"] == 4
    assert rows[("rag_verify", "cases")]["verifier_support_rate"] == 1.0
    assert rows[("rag", "cases")]["citation_validity"] == 1.0
    assert report["taxonomy"]["codes"] > 0
    dev_cost = json.loads((tmp_path / "dev.json").read_text())["total_cost_usd"]
    assert report["total_cost_usd"] + dev_cost == pytest.approx(factory.calls * 0.001)
    md = (tmp_path / "r.md").read_text()
    assert "| rag_verify | claude-haiku-5-5 | 0.05 | 4 |" in md
    # rag_verify answered every case with a tuned cut-off, so the audit sample exists.
    audit = read_audit(tmp_path / "audit.csv")
    assert 0 < len(audit) <= 50 and all(r["grade"] == "" for r in audit)

    # Resume: nothing left to do, no LLM call.
    before = factory.calls
    assert _main(tmp_path, db_url, factory, "--workers", "2") == 0
    assert factory.calls == before
    assert read_audit(tmp_path / "audit.csv") == audit  # never resampled


def test_stop_on_unavailable_provider_then_resume(
    indexed: Connection, db_url: str, tmp_path: Path
) -> None:
    failing = CountingFactory(Responder(), fail_after=2)
    args = ("--cases", "3", "--systems", "closed_book", "--workers", "1")
    assert _main(tmp_path, db_url, failing, *args) == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert "ProviderUnavailableError" in report["stopped"]
    row = next(r for r in report["rows"] if r["system"] == "closed_book")
    assert (row["n_cases"], row["complete"]) == (2, False)
    resumed = CountingFactory(Responder())
    assert _main(tmp_path, db_url, resumed, *args) == 0
    assert resumed.calls == 1  # only the missing case
    row = next(
        r
        for r in json.loads((tmp_path / "r.json").read_text())["rows"]
        if r["system"] == "closed_book"
    )
    assert row["complete"]

    failing = CountingFactory(Responder(), fail_after=1)
    assert (
        _main(tmp_path, db_url, failing, "--cases", "3", "--systems", "rag", "--workers", "1") == 1
    )
    done = ce.ResultCache(tmp_path / "runs").load("rag", "claude-haiku-5-5")
    assert len(done) == 0  # the case that failed mid-run is not cached
    good = CountingFactory(Responder())
    assert _main(tmp_path, db_url, good, "--cases", "3", "--systems", "rag") == 0
    assert len(ce.ResultCache(tmp_path / "runs").load("rag", "claude-haiku-5-5")) == 3


def test_budget_stop(
    indexed: Connection, db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_USD", "0.0025")
    factory = CountingFactory(Responder(), usd=0.001)
    code = _main(
        tmp_path, db_url, factory, "--cases", "6", "--systems", "closed_book", "--workers", "1"
    )
    assert code == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert "BudgetExceededError" in report["stopped"]
    assert factory.calls == 3  # 0.000, 0.001, 0.002 < 0.0025; the fourth is refused


def test_limit_repeat_and_extra_model(indexed: Connection, db_url: str, tmp_path: Path) -> None:
    factory = CountingFactory(Responder())
    base = ("--cases", "3", "--systems", "closed_book", "--workers", "2")
    assert _main(tmp_path, db_url, factory, *base, "--limit", "2") == 0
    row = next(
        r
        for r in json.loads((tmp_path / "r.json").read_text())["rows"]
        if r["system"] == "closed_book"
    )
    assert (row["n_cases"], row["complete"]) == (2, False)

    assert _main(tmp_path, db_url, factory, *base, "--repeat", "2") == 0
    row = next(
        r
        for r in json.loads((tmp_path / "r.json").read_text())["rows"]
        if r["system"] == "closed_book"
    )
    assert row["repeats"] == 2 and row["complete"] and "micro_f1_min" in row

    assert _main(tmp_path, db_url, factory, *base, "--model", "claude-sonnet-5-5") == 0
    models = {
        r["model"]
        for r in json.loads((tmp_path / "r.json").read_text())["rows"]
        if r["system"] == "closed_book"
    }
    assert models == {"claude-haiku-5-5", "claude-sonnet-5-5"}


def test_freeze_cases(tmp_path: Path) -> None:
    scored = [f"e{i:03d}" for i in range(20)]
    path = tmp_path / "cases.json"
    first = ce.freeze_cases(path, scored, 5, seed=1)
    assert len(first) == 5 and ce.freeze_cases(path, scored, None, seed=9) == first
    with pytest.raises(SystemExit):
        ce.freeze_cases(path, scored, 6, seed=1)
    assert len(ce.freeze_cases(path, scored, 6, seed=1, refreeze=True)) == 6
    assert ce.freeze_cases(tmp_path / "all.json", scored, None, seed=1) == scored


def test_audit_sample_and_grading(tmp_path: Path) -> None:
    answers = [
        {
            "ev_id": f"t{i}",
            "findings": [
                {
                    "label": A,
                    "rationale": "why",
                    "citations": [f"h{i}:analysis:0", f"h{i}:factual:0"],
                }
            ],
        }
        for i in range(40)
    ]
    path = tmp_path / "audit.csv"
    assert write_audit_sample(path, answers, {A: "Name"}, seed=3)
    rows = read_audit(path)
    assert len(rows) == 50 and len({(r["ev_id"], r["passage_id"]) for r in rows}) == 50
    assert not write_audit_sample(path, answers, {A: "Name"}, seed=4)  # never overwritten

    answers_iter = iter(["2", "x", "1", "s", "0", "q"])
    shown: list[str] = []
    n = grade_loop(
        path,
        lambda ev: f"incident {ev}",
        lambda pid: f"text of {pid}",
        ask=lambda _: next(answers_iter),
        show=shown.append,
    )
    assert n == 3
    assert any("text of" in s for s in shown)
    grades = [r["grade"] for r in read_audit(path)]
    assert grades[:4] == ["2", "1", "", "0"]
    summary = audit_summary(path, 0.9)
    assert summary is not None
    assert summary["graded"] == 3
    assert summary["precision_strict"] == pytest.approx(1 / 3)
    assert summary["precision_lenient"] == pytest.approx(2 / 3)
    assert summary["verifier_support_rate"] == 0.9

    # Resume: the skipped row comes first next time.
    assert (
        grade_loop(path, lambda ev: "", lambda pid: "", ask=lambda _: "2", show=lambda _: None)
        == 47
    )


def test_shared_models_use_one_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    # The embedder and reranker share the accelerator; on Apple GPUs (MPS) two threads
    # encoding work at once abort the process, so both wrappers must hold the same lock.
    monkeypatch.setattr(ce, "default_embedder", FakeEmbedder)
    monkeypatch.setattr(ce, "default_reranker", FakeReranker)
    make = ce.default_retriever_factory()
    first, second = make(None), make(None)  # type: ignore[arg-type]
    embedder, reranker = first._embedder, first._reranker
    assert isinstance(embedder, ce.LockedEmbedder)
    assert isinstance(reranker, ce.LockedReranker)
    assert embedder._lock is reranker._lock
    assert second._embedder is embedder and second._reranker is reranker


# --- Dev split and confidence cut-offs --------------------------------------------------


def test_apply_cutoff_and_tune_confidence_on_hand_made_confidences() -> None:
    def res(ev: str, *found: tuple[str, float]) -> dict[str, Any]:
        findings = [
            {"label": lbl, "confidence": c, "citations": [f"{lbl}:p"], "rationale": "r"}
            for lbl, c in found
        ]
        return {
            "ev_id": ev,
            "labels": [lbl for lbl, _ in found],
            "abstained": not found,
            "cites": True,
            "findings": findings,
            "citations": [{"label": lbl, "passage_id": f"{lbl}:p"} for lbl, _ in found],
        }

    gold = {"e1": frozenset({A}), "e2": frozenset({B}), "e3": frozenset({C})}
    results = [
        res("e1", (A, 0.9), (B, 0.4)),  # right label high, wrong one low
        res("e2", (B, 0.6), (D, 0.3)),
        res("e3", (C, 0.5), (A, 0.5), (D, 0.1)),  # a wrong label as confident as the right
    ]
    cut = apply_cutoff(results[0], 0.5)
    assert cut["labels"] == [A] and cut["citations"] == [{"label": A, "passage_id": f"{A}:p"}]
    assert cut["cutoff"] == 0.5 and [f["label"] for f in cut["findings"]] == [A]
    assert apply_cutoff(results[0], 0.95)["abstained"] is True
    assert apply_cutoff(results[0], None) == results[0]  # untuned: unchanged
    assert apply_cutoff({"ev_id": "e", "labels": [], "abstained": True}, 0.5)["labels"] == []
    assert apply_cutoff(results[2], 0.5)["labels"] == [C, A]  # confidence >= tau is kept

    tau, scores = tune_confidence(results, gold)
    # F1 = 2tp / (2tp + fp + fn). tau <= 0.1: tp 3, fp 4 -> 6/10; 0.15-0.3: fp 3 -> 6/9;
    # 0.35-0.4: fp 2 -> 6/8; 0.45-0.5: fp 1 -> 6/7; 0.55-0.6: tp 2, fn 1 -> 4/5;
    # 0.65-0.9: tp 1, fn 2 -> 2/4; 0.95: nothing left.
    assert scores[0.05] == pytest.approx(6 / 10) and scores[0.3] == pytest.approx(6 / 9)
    assert scores[0.45] == pytest.approx(6 / 7) and scores[0.5] == pytest.approx(6 / 7)
    assert scores[0.55] == pytest.approx(4 / 5) and scores[0.9] == pytest.approx(2 / 4)
    assert scores[0.95] == 0.0
    assert tau == 0.45  # best F1, tied with 0.5: ties go to the lower cut-off


class GoldResponder(Responder):
    """closed_book: the case's gold label at 0.7 and two wrong labels at 0.3 and 0.2.

    On any set of cases the best cut-off is then 0.35 (keep only the gold label).
    """

    def __init__(self, gold_by_date: dict[str, list[str]], codes: list[str]) -> None:
        super().__init__()
        self.gold_by_date = gold_by_date
        self.codes = codes

    def __call__(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if not prompt.startswith("# Task: closed_book"):
            return super().__call__(system, prompt, schema)
        day = re.search(r"^Date: (\S+)$", prompt, re.M)
        assert day is not None
        right = self.gold_by_date[day.group(1)][0]
        wrong = [c for c in self.codes if c not in self.gold_by_date[day.group(1)]][:2]
        return {
            "findings": [
                {"label": right, "confidence": 0.7, "rationale": "r"},
                {"label": wrong[0], "confidence": 0.3, "rationale": "r"},
                {"label": wrong[1], "confidence": 0.2, "rationale": "r"},
            ]
        }


class RecordingRetriever(Retriever):
    """Records every search's cutoff and excluded event."""

    log: ClassVar[list[tuple[str, Any, Any]]] = []

    def search(self, query: str, top_k: int = 10, **kw: Any) -> Any:
        RecordingRetriever.log.append(("search", kw["before_date"], kw.get("exclude_event_id")))
        return super().search(query, top_k, **kw)

    def search_events(self, query: str, top_k: int = 10, **kw: Any) -> Any:
        RecordingRetriever.log.append(("events", kw["before_date"], kw.get("exclude_event_id")))
        return super().search_events(query, top_k, **kw)


def test_dev_split_is_history_only_and_respects_each_cutoff(
    indexed: Connection, db_url: str, tmp_path: Path
) -> None:
    dates: dict[str, date] = dict(indexed.execute("SELECT ev_id, ev_date FROM events").fetchall())
    tuning = ce.tuning_events(indexed, None, 42)
    assert tuning and all(date(2023, 1, 1) <= dates[ev] < HELD_OUT_FROM for ev in tuning)

    RecordingRetriever.log = []
    factory = CountingFactory(Responder())
    code = _main(
        tmp_path,
        db_url,
        factory,
        "--split",
        "dev",
        "--systems",
        "knn_vote",
        "random_context",
        "rag",
        "rag_verify",
        retriever_factory=lambda c: RecordingRetriever(c, FakeEmbedder(), FakeReranker()),
    )
    assert code == 0
    frozen = json.loads((tmp_path / "dev_cases.json").read_text())
    dev = frozen["ev_ids"]
    assert sorted(dev) == sorted(tuning)  # default: every 2023 tuning event
    assert all(dates[ev] < HELD_OUT_FROM for ev in dev)  # never a held-out event

    # Every search (kNN tuning and the agents) is for a 2023 event, before its own date.
    assert RecordingRetriever.log
    for _, before, exclude in RecordingRetriever.log:
        assert exclude in tuning and before == corpus_cutoff(dates[exclude]) == dates[exclude]

    # Every context passage is from an earlier event, never the case itself.
    runs = ce.ResultCache(tmp_path / "runs" / "dev")
    passage_event: dict[str, str] = dict(
        indexed.execute("SELECT passage_id, event_id FROM passages").fetchall()
    )
    for system in ("random_context", "rag", "rag_verify"):
        records = list(runs.load(system, "claude-haiku-5-5").values())
        assert {r["ev_id"] for r in records} == set(dev)
        for r in records:
            for pid in r["context_passages"]:
                ev = passage_event[pid]
                assert ev != r["ev_id"] and dates[ev] < dates[r["ev_id"]]
    # Dev results stay out of the held-out cache and the held-out report.
    assert not list((tmp_path / "runs").glob("rag*.jsonl"))
    assert not (tmp_path / "r.json").exists()
    report = json.loads((tmp_path / "dev.json").read_text())
    assert report["split"] == "dev" and report["cases"]["n"] == len(dev)
    rows = {(r["system"], r["cutoff"] is None) for r in report["rows"]}
    assert ("rag", True) in rows and ("rag", False) in rows
    assert ("knn_vote", True) in rows and "closed_book" not in {s for s, _ in rows}


def test_held_out_scoring_applies_the_dev_tuned_cutoff(
    indexed: Connection, db_url: str, tmp_path: Path
) -> None:
    gold = get_all_gold_labels(indexed)
    dates: dict[str, date] = dict(indexed.execute("SELECT ev_id, ev_date FROM events").fetchall())
    by_date: dict[str, list[str]] = {}
    for ev in sorted(gold):
        by_date.setdefault(dates[ev].isoformat(), []).extend(sorted(gold[ev]))
    codes = sorted(load_taxonomy(indexed))
    factory = CountingFactory(GoldResponder(by_date, codes))
    args = ("--systems", "closed_book", "--workers", "2")

    # Held-out first: no dev results yet, so no cut-off and every finding counts.
    assert _main(tmp_path, db_url, factory, "--cases", "4", *args) == 0
    row = next(r for r in json.loads((tmp_path / "r.json").read_text())["rows"])
    assert row["system"] == "closed_book" and row["cutoff"] is None
    assert row["predicted"] == 3 * 4

    assert _main(tmp_path, db_url, factory, "--split", "dev", "--cases", "4", *args) == 0
    dev_report = json.loads((tmp_path / "dev.json").read_text())
    tuned = dev_report["tuning"]["llm_cutoffs"]["closed_book__claude-haiku-5-5"]
    assert tuned["tau"] == 0.35 and tuned["dev_cases"] == 4
    assert tuned["dev_micro_f1"] > tuned["dev_micro_f1_raw"]
    assert "| closed_book | claude-haiku-5-5 | 0.35 | 4 |" in (tmp_path / "dev.md").read_text()

    # The held-out report re-scores the cached results with the tuned cut-off: no new calls.
    calls = factory.calls
    assert _main(tmp_path, db_url, factory, "--cases", "4", *args) == 0
    assert factory.calls == calls
    report = json.loads((tmp_path / "r.json").read_text())
    row = next(r for r in report["rows"] if r["system"] == "closed_book")
    assert row["cutoff"] == 0.35 and row["predicted"] == 4  # only the 0.7 findings remain
    assert report["tuning"]["llm_cutoffs"]["closed_book__claude-haiku-5-5"]["tau"] == 0.35
    records = ce.ResultCache(tmp_path / "runs").load("closed_book", "claude-haiku-5-5").values()
    expected = score_cases(
        [apply_cutoff(r, 0.35) for r in records],
        gold,
        history_gold={},
        passages={},
        cutoffs={r["ev_id"]: HELD_OUT_FROM for r in records},
    )
    assert row["micro_f1"] == pytest.approx(expected["micro_f1"])
    md = (tmp_path / "r.md").read_text()
    assert "closed_book__claude-haiku-5-5: cut-off 0.35" in md
