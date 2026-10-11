"""The multi-hop agent with FakeProvider and fake retrieval models (no network, no LLM)."""

import ast
import re
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
import pytest
from conftest import FIXTURES, reset
from fakes import FakeEmbedder, FakeReranker

import causetrace.agent
from causetrace.agent import Agent
from causetrace.agent.prompts import PROMPT_NAMES
from causetrace.config import AGENT_LIMITS, AgentLimits
from causetrace.db import Connection
from causetrace.index import build_index
from causetrace.ingest import ingest
from causetrace.llm import LLMResponseError
from causetrace.llm.fake import FakeProvider
from causetrace.retrieval import Retriever
from causetrace.split import HELD_OUT_FROM

PASSAGE_ID = re.compile(r"^\[([^\]]+)\] ", re.M)
CODE = re.compile(r"^(\d{2}-\d{2}-\d{2})  ", re.M)
PAIR_FINDING = re.compile(r"^Finding (\d{2}-\d{2}-\d{2}) ", re.M)
PAIR_PASSAGE = re.compile(r"passage \[([^\]]+)\]")

Handler = Callable[[str, str], dict[str, Any]]


@pytest.fixture(scope="module")
def indexed(db_url: str) -> Iterator[Connection]:
    with psycopg.connect(db_url, autocommit=True) as conn:
        reset(conn)
        ingest(conn, FIXTURES)
        build_index(conn, FakeEmbedder(), batch_size=256)
        yield conn


def _held_out_event(conn: Connection) -> str:
    row = conn.execute(
        "SELECT ev_id FROM events WHERE ev_date >= %s ORDER BY ev_date DESC, ev_id LIMIT 1",
        (HELD_OUT_FROM,),
    ).fetchone()
    assert row is not None
    return str(row[0])


def _task(prompt: str) -> str:
    return prompt.splitlines()[0].removeprefix("# Task: ").strip()


class Script:
    """Answers each node from a handler; handlers can be swapped per test."""

    def __init__(self) -> None:
        self.handlers: dict[str, Handler] = {
            "plan": lambda s, p: {"queries": ["loss of engine power after takeoff",
                                              "pilot decision to continue flight"]},
            "check_evidence": lambda s, p: {"enough": True, "missing": "", "gap_query": ""},
            "analyze": self.good_findings,
            "verify": self.verdicts("SUPPORTED"),
        }  # fmt: skip
        self.calls: dict[str, int] = {}

    def __call__(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        task = _task(prompt)
        self.calls[task] = self.calls.get(task, 0) + 1
        return self.handlers[task](system, prompt)

    @staticmethod
    def good_findings(system: str, prompt: str, n: int = 2) -> dict[str, Any]:
        codes = CODE.findall(system)
        pids = PASSAGE_ID.findall(prompt)
        return {
            "findings": [
                {"label": codes[i], "confidence": 0.9 - i / 10, "citations": [pids[i]],
                 "rationale": "similar sequence"}
                for i in range(n)
            ]
        }  # fmt: skip

    @staticmethod
    def verdicts(verdict: str) -> Handler:
        def handler(system: str, prompt: str) -> dict[str, Any]:
            out = []
            for block in prompt.split("Finding ")[1:]:
                label = block.split(" ", 1)[0]
                for pid in PAIR_PASSAGE.findall(block):
                    out.append(
                        {"label": label, "passage_id": pid, "verdict": verdict, "reason": "x"}
                    )
            return {"verdicts": out}

        return handler


def _agent(
    conn: Connection, script: Script, limits: AgentLimits = AGENT_LIMITS
) -> tuple[Agent, FakeProvider]:
    provider = FakeProvider(script)
    retriever = Retriever(conn, FakeEmbedder(), FakeReranker())
    return Agent(conn, retriever, provider, limits=limits), provider


def _nodes(result: dict[str, Any]) -> list[str]:
    return [t["node"] for t in result["trace"]]


def test_happy_path(indexed: Connection) -> None:
    script = Script()
    agent, provider = _agent(indexed, script)
    ev_id = _held_out_event(indexed)
    result = agent.run(ev_id)

    assert result["abstained"] is False
    assert len(result["findings"]) == 2
    assert result["hops"] == 0 and result["revisions"] == 0 and result["evidence_ok"] is True
    assert _nodes(result) == [
        "plan",
        "plan",
        "retrieve",
        "check_evidence",
        "analyze",
        "verify",
        "answer",
    ]
    retrieved = {p["passage_id"] for p in result["passages"]}
    for f in result["findings"]:
        assert f["label"] in agent.taxonomy and set(f["citations"]) <= retrieved
        assert f["short_name"] == agent.taxonomy[f["label"]]
    # Effort per node and caching of the fixed system prompt + taxonomy on every call.
    efforts = {_task(c["prompt"]): c["effort"] for c in provider.calls}
    assert efforts == {
        "plan": "low",
        "check_evidence": "low",
        "analyze": "medium",
        "verify": "medium",
    }
    assert all(c["cache_system"] for c in provider.calls)
    assert set(result["prompts"]) == set(PROMPT_NAMES)
    assert all(len(h) == 64 for h in result["prompts"].values())
    assert result["usage"]["calls"] == 4
    assert result["taxonomy"]["codes"] == len(agent.taxonomy) > 0


def test_taxonomy_is_history_only(indexed: Connection) -> None:
    agent, _ = _agent(indexed, Script())
    history_codes = {
        r[0]
        for r in indexed.execute(
            "SELECT DISTINCT f.label_code FROM findings f JOIN events e USING (ev_id)"
            " WHERE f.cm_inpc AND e.ev_date < %s",
            (HELD_OUT_FROM,),
        )
    }
    assert set(agent.taxonomy) == history_codes


def test_planned_queries_use_rerank_and_packet_fallback_uses_rrf(indexed: Connection) -> None:
    script = Script()
    agent, _ = _agent(indexed, script)
    result = agent.run(_held_out_event(indexed))
    searches = next(t for t in result["trace"] if t["node"] == "retrieve")["searches"]
    assert [s["mode"] for s in searches] == ["rrf_rerank", "rrf_rerank"]

    def invalid(system: str, prompt: str) -> dict[str, Any]:
        raise LLMResponseError("invalid JSON from model")

    script.handlers["plan"] = invalid
    result = agent.run(_held_out_event(indexed))
    plan_summary = [t for t in result["trace"] if t["node"] == "plan"][-1]
    assert plan_summary["fallback_to_packet"] is True
    query = result["queries"][0]
    searches = next(t for t in result["trace"] if t["node"] == "retrieve")["searches"]
    expected = "rrf_rerank" if len(query.split()) < AGENT_LIMITS.rerank_max_words else "rrf"
    assert searches[0]["mode"] == expected
    assert len(query.split()) >= AGENT_LIMITS.rerank_max_words  # a narrative, so rrf
    assert searches[0]["mode"] == "rrf"


def test_invalid_json_is_retried_once(indexed: Connection) -> None:
    script = Script()
    attempts = {"n": 0}

    def flaky(system: str, prompt: str) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LLMResponseError("invalid JSON from model")
        return {"queries": ["carburetor icing in humid conditions"]}

    script.handlers["plan"] = flaky
    agent, _ = _agent(indexed, script)
    result = agent.run(_held_out_event(indexed))
    plans = [t for t in result["trace"] if t["node"] == "plan"]
    assert [p.get("attempt") for p in plans[:2]] == [1, 2]
    assert "error" in plans[0] and "error" not in plans[1]
    assert result["queries"] == ["carburetor icing in humid conditions"]
    assert plans[-1]["fallback_to_packet"] is False

    # Two invalid answers: no third try, fall back to the packet text.
    script.handlers["plan"] = lambda s, p: (_ for _ in ()).throw(LLMResponseError("bad"))
    script.calls.clear()
    result = agent.run(_held_out_event(indexed))
    assert script.calls["plan"] == 2
    assert [t for t in result["trace"] if t["node"] == "plan"][-1]["fallback_to_packet"] is True


def test_extra_hop_path(indexed: Connection) -> None:
    script = Script()
    script.handlers["check_evidence"] = lambda s, p: {
        "enough": False,
        "missing": "fuel system precedent",
        "gap_query": "fuel exhaustion",
    }
    agent, _ = _agent(indexed, script)
    result = agent.run(_held_out_event(indexed))
    assert result["hops"] == 1
    assert result["evidence_ok"] is False
    retrieves = [t for t in result["trace"] if t["node"] == "retrieve"]
    assert len(retrieves) == 2 and retrieves[1]["hop"] is True
    assert [s["query"] for s in retrieves[1]["searches"]] == ["fuel exhaustion"]
    assert script.calls["check_evidence"] == 1  # at most one extra hop, no second check
    assert _nodes(result)[-3:] == ["analyze", "verify", "answer"]
    assert len(result["passages"]) <= AGENT_LIMITS.max_passages


def test_revision_path(indexed: Connection) -> None:
    script = Script()
    first = {"done": False}

    def analyze(system: str, prompt: str) -> dict[str, Any]:
        if not first["done"]:
            first["done"] = True
            return {"findings": [{"label": "99-99-99", "confidence": 0.8,
                                  "citations": ["20990101X00000:analysis:0"],
                                  "rationale": "?"}]}  # fmt: skip
        assert "not in the taxonomy" in prompt and "was not retrieved" in prompt
        return Script.good_findings(system, prompt, n=1)

    script.handlers["analyze"] = analyze
    agent, _ = _agent(indexed, script)
    result = agent.run(_held_out_event(indexed))
    assert result["revisions"] == 1
    assert script.calls["analyze"] == 2
    assert result["abstained"] is False and len(result["findings"]) == 1
    assert _nodes(result)[-5:] == ["analyze", "verify", "analyze", "verify", "answer"]


def test_abstain_path(indexed: Connection) -> None:
    script = Script()
    script.handlers["verify"] = Script.verdicts("UNSUPPORTED")
    agent, _ = _agent(indexed, script)
    result = agent.run(_held_out_event(indexed))
    assert result["abstained"] is True and result["findings"] == []
    assert result["revisions"] == 1  # revised once, then abstained
    assert script.calls["analyze"] == 2 and script.calls["verify"] == 2
    assert result["trace"][-1] == {"node": "abstain", "reason": "no supported finding"}


def test_partial_support_is_not_an_answer(indexed: Connection) -> None:
    script = Script()
    script.handlers["verify"] = Script.verdicts("PARTIAL")
    agent, _ = _agent(indexed, script)
    assert agent.run(_held_out_event(indexed))["abstained"] is True


def test_limits_are_enforced(indexed: Connection) -> None:
    script = Script()
    script.handlers["plan"] = lambda s, p: {
        "queries": [f"query number {i} about icing" for i in range(6)]
    }
    script.handlers["check_evidence"] = lambda s, p: {"enough": False, "missing": "more",
                                                      "gap_query": "more evidence"}  # fmt: skip
    script.handlers["analyze"] = lambda s, p: Script.good_findings(s, p, n=8)
    script.handlers["verify"] = Script.verdicts("UNSUPPORTED")  # always asks for a revision
    agent, provider = _agent(indexed, script)
    result = agent.run(_held_out_event(indexed))
    assert len(result["queries"]) == 3
    first = next(t for t in result["trace"] if t["node"] == "retrieve")
    assert len(first["searches"]) == 3
    assert result["hops"] == 1 and result["revisions"] == 1
    assert len(result["passages"]) <= 10
    assert all(len(p) <= 5 for p in [result["predictions"]])
    analyzes = [t for t in result["trace"] if t["node"] == "analyze"]
    assert analyzes[0]["returned"] == 8 and analyzes[0]["kept"] == 5
    # The longest possible run: 1 plan + 1 check + 2 analyze + 2 verify calls.
    assert result["usage"]["calls"] == 6 == len(provider.calls)


def test_token_ceiling_stops_llm_calls(indexed: Connection) -> None:
    script = Script()
    agent, provider = _agent(indexed, script, replace(AGENT_LIMITS, max_tokens=50))
    result = agent.run(_held_out_event(indexed))
    assert len(provider.calls) == 1  # only plan; every later call is skipped
    assert result["abstained"] is True
    skipped = [t for t in result["trace"] if t.get("skipped") == "token ceiling reached"]
    assert [t["node"] for t in skipped] == ["analyze"]


def test_target_event_is_never_retrieved(indexed: Connection) -> None:
    events = indexed.execute(
        "(SELECT ev_id, ev_date FROM events WHERE ev_date >= %s ORDER BY ev_id LIMIT 3)"
        " UNION ALL"
        " (SELECT ev_id, ev_date FROM events WHERE ev_date < %s ORDER BY ev_id DESC LIMIT 3)",
        (HELD_OUT_FROM, HELD_OUT_FROM),
    ).fetchall()
    script = Script()
    script.handlers["check_evidence"] = lambda s, p: {"enough": False, "missing": "x",
                                                      "gap_query": "engine failure"}  # fmt: skip
    agent, _ = _agent(indexed, script)
    dates: dict[str, date] = dict(indexed.execute("SELECT ev_id, ev_date FROM events").fetchall())
    for ev_id, ev_date in events:
        # The target's own words as the query: the strongest possible self-match.
        own = indexed.execute(
            "SELECT content FROM passages WHERE event_id = %s ORDER BY passage_id LIMIT 1", (ev_id,)
        ).fetchone()
        assert own is not None
        query = " ".join(str(own[0]).split()[:40])

        def plan(system: str, prompt: str, query: str = query) -> dict[str, Any]:
            return {"queries": [query]}

        script.handlers["plan"] = plan
        result = agent.run(ev_id)
        cutoff = min(ev_date, HELD_OUT_FROM)
        assert result["passages"]
        for p in result["passages"]:
            assert p["event_id"] != ev_id
            assert dates[p["event_id"]] < cutoff
        for t in result["trace"]:
            if t["node"] == "retrieve":
                assert all(not pid.startswith(ev_id + ":") for pid in t["passages"])


def test_agent_never_imports_gold() -> None:
    package = Path(causetrace.agent.__file__).parent
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            assert not any("gold" in n for n in names), (path.name, names)
        source = path.read_text(encoding="utf-8")
        for column in ("narr_accf", "narr_cause", "narr_accp_safe"):
            assert column not in source, (path.name, column)
