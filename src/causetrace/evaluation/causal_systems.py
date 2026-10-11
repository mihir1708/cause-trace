"""The six causal systems of docs/EVAL_PROTOCOL.md, each mapping one held-out event to a
predicted label set (with citations for the systems that cite).

All systems see only the target packet (``causetrace.packets``) and history: retrieval and
random passages use ``corpus_cutoff`` and exclude the target. Gold labels are read only for
history events (frequencies, neighbour votes, tuning), never for the target.
"""

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import date
from typing import Any, Protocol

from causetrace.agent import Agent
from causetrace.agent.graph import describe_packet
from causetrace.agent.prompts import load_prompt, load_prompts
from causetrace.agent.taxonomy import render_taxonomy, taxonomy_sha256
from causetrace.config import AGENT_LIMITS
from causetrace.db import Connection
from causetrace.evaluation.retrieval_eval import packet_query_text
from causetrace.llm import LLMResponseError, Provider
from causetrace.packets import get_incident_packet
from causetrace.retrieval import Mode, Retriever
from causetrace.split import corpus_cutoff

SYSTEMS = ("frequency_prior", "knn_vote", "closed_book", "random_context", "rag", "rag_verify")
FREE_SYSTEMS = ("frequency_prior", "knn_vote")
LLM_SYSTEMS = ("closed_book", "random_context", "rag", "rag_verify")
KNN_K = 10
KNN_MODE: Mode = "rrf"  # best held-out Label Recall@10 for narrative queries
TAU_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))
FREQ_K_GRID = (1, 2, 3, 4, 5)
RANDOM_PASSAGES = 10


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def micro_f1(pairs: Sequence[tuple[set[str], frozenset[str]]]) -> float:
    tp = sum(len(p & g) for p, g in pairs)
    fp = sum(len(p - g) for p, g in pairs)
    fn = sum(len(g - p) for p, g in pairs)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


class System(Protocol):
    name: str
    model: str  # "none" for systems without an LLM

    def config(self) -> dict[str, Any]:
        """Everything that changes the output; hashed into the cache key."""
        ...

    def predict(self, conn: Connection, ev_id: str) -> dict[str, Any]: ...


# --- Free systems ---------------------------------------------------------------------


class FrequencyPrior:
    """The k labels most often found in history events; k tuned on 2023 events."""

    name = "frequency_prior"
    model = "none"

    def __init__(
        self,
        history_gold: Mapping[str, frozenset[str]],
        dates: Mapping[str, date],
        tuning_events: Sequence[str],
        cutoff: date,
        tuning_start: date,
    ) -> None:
        pre_tuning = Counter(
            lbl for ev, g in history_gold.items() if dates[ev] < tuning_start for lbl in g
        )
        tuning = [(ev, history_gold[ev]) for ev in tuning_events if ev in history_gold]
        scores = {
            k: micro_f1([({lbl for lbl, _ in _top(pre_tuning, k)}, g) for _, g in tuning])
            for k in FREQ_K_GRID
        }
        self.k = max(FREQ_K_GRID, key=lambda k: (scores[k], -k))
        self.tuning_f1 = scores
        counts = Counter(lbl for ev, g in history_gold.items() if dates[ev] < cutoff for lbl in g)
        self.labels = [lbl for lbl, _ in _top(counts, self.k)]

    def config(self) -> dict[str, Any]:
        return {"k": self.k, "labels": self.labels}

    def predict(self, conn: Connection, ev_id: str) -> dict[str, Any]:
        return {"labels": list(self.labels), "abstained": False, "cites": False}


def _top(counts: Counter[str], k: int) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]


def knn_scores(neighbours: Sequence[str], gold: Mapping[str, frozenset[str]]) -> dict[str, float]:
    """Rank-weighted label votes: weight 1/rank, normalized by the total weight."""
    weights = [1.0 / r for r in range(1, len(neighbours) + 1)]
    total = sum(weights)
    scores: dict[str, float] = {}
    for w, ev in zip(weights, neighbours, strict=True):
        for lbl in gold.get(ev, ()):
            scores[lbl] = scores.get(lbl, 0.0) + w / total
    return scores


def knn_predict(
    neighbours: Sequence[str], gold: Mapping[str, frozenset[str]], tau: float, limit: int
) -> list[str]:
    scores = knn_scores(neighbours, gold)
    keep = [lbl for lbl, s in scores.items() if s >= tau - 1e-12]
    return sorted(keep, key=lambda lbl: (-scores[lbl], lbl))[:limit]


def tune_tau(
    neighbour_lists: Mapping[str, Sequence[str]],
    gold: Mapping[str, frozenset[str]],
    limit: int,
) -> tuple[float, dict[float, float]]:
    """The vote cut-off with the best micro F1 on the tuning events (ties: lowest)."""
    scores = {
        tau: micro_f1(
            [(set(knn_predict(n, gold, tau, limit)), gold[ev]) for ev, n in neighbour_lists.items()]
        )
        for tau in TAU_GRID
    }
    best = max(TAU_GRID, key=lambda t: (scores[t], -t))
    return best, scores


def knn_neighbours(retriever: Retriever, conn: Connection, ev_id: str) -> list[str]:
    """Top KNN_K distinct history events for the event's packet, before its own cutoff."""
    packet = get_incident_packet(ev_id, conn)
    result = retriever.search_events(
        packet_query_text(packet),
        KNN_K,
        before_date=corpus_cutoff(packet.ev_date),
        exclude_event_id=ev_id,
        mode=KNN_MODE,
    )
    return [e.event_id for e in result.events]


class KnnVote:
    """Labels voted by the top 10 retrieved history events (cut-off tuned on 2023)."""

    name = "knn_vote"
    model = "none"

    def __init__(
        self,
        history_gold: Mapping[str, frozenset[str]],
        tau: float,
        neighbours: Callable[[Connection, str], list[str]],
        retrieval_config: Mapping[str, Any],
    ) -> None:
        self.history_gold = history_gold
        self.tau = tau
        self.neighbours = neighbours
        self.retrieval_config = dict(retrieval_config)

    def config(self) -> dict[str, Any]:
        return {"tau": self.tau, "k": KNN_K, "mode": KNN_MODE, **self.retrieval_config}

    def predict(self, conn: Connection, ev_id: str) -> dict[str, Any]:
        start = time.perf_counter()
        neighbours = self.neighbours(conn, ev_id)
        labels = knn_predict(neighbours, self.history_gold, self.tau, AGENT_LIMITS.max_findings)
        return {
            "labels": labels,
            "abstained": not labels,
            "cites": False,
            "neighbours": neighbours,
            "seconds": time.perf_counter() - start,
        }


# --- LLM systems ----------------------------------------------------------------------

CLOSED_BOOK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["label", "confidence", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


def _ranked_findings(findings: Any, limit: int) -> list[dict[str, Any]]:
    items = [f for f in findings if isinstance(f, dict)] if isinstance(findings, list) else []

    def conf(f: dict[str, Any]) -> float:
        c = f.get("confidence")
        return float(c) if isinstance(c, int | float) else 0.0

    return sorted(items, key=lambda f: -conf(f))[:limit]


def _labels(findings: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(f.get("label", "")).strip() for f in findings if f.get("label")))


def _trace_usage(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    llm = [t for t in trace if "model" in t]
    return {
        "calls": len(llm),
        "tokens": sum(int(t.get("tokens", 0)) for t in llm),
        "cost_usd": sum(float(t.get("cost_usd", 0.0)) for t in llm),
    }


class ClosedBook:
    """The LLM with the packet and the taxonomy only."""

    name = "closed_book"

    def __init__(self, provider: Provider, taxonomy: Mapping[str, str]) -> None:
        self.provider = provider
        self.model = provider.model
        self.taxonomy = dict(taxonomy)
        self.prompt = load_prompt("closed_book")
        self.system_prompt = load_prompt("system")
        self.system_blocks = [self.system_prompt.text, render_taxonomy(self.taxonomy)]

    def config(self) -> dict[str, Any]:
        return {
            "prompts": {"system": self.system_prompt.sha256, "closed_book": self.prompt.sha256},
            "taxonomy": taxonomy_sha256(self.taxonomy),
            "effort": "medium",
        }

    def predict(self, conn: Connection, ev_id: str) -> dict[str, Any]:
        start = time.perf_counter()
        packet = get_incident_packet(ev_id, conn)
        prompt = self.prompt.render(
            packet=describe_packet(packet), max_findings=AGENT_LIMITS.max_findings
        )
        try:
            parsed, usage = self.provider.generate_json(
                self.system_blocks, prompt, CLOSED_BOOK_SCHEMA, effort="medium", cache_system=True
            )
        except LLMResponseError as err:
            return {
                "labels": [],
                "abstained": True,
                "cites": False,
                "error": str(err),
                "calls": 1,
                "seconds": time.perf_counter() - start,
            }
        findings = _ranked_findings(parsed.get("findings"), AGENT_LIMITS.max_findings)
        labels = _labels(findings)
        return {
            "labels": labels,
            "abstained": not labels,
            "cites": False,
            "findings": findings,
            "calls": 1,
            "tokens": usage.input_tokens
            + usage.output_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens,
            "cost_usd": usage.cost_usd,
            "seconds": time.perf_counter() - start,
        }


_RANDOM_SQL = """
SELECT passage_id, event_id, event_date, section_type, content
FROM passages
WHERE event_date < %(before)s AND event_id <> %(ev_id)s
ORDER BY md5(passage_id || %(salt)s)
LIMIT %(n)s
"""


def random_passages(conn: Connection, ev_id: str, before: date, seed: int) -> list[dict[str, Any]]:
    """RANDOM_PASSAGES history passages, a fixed pseudo-random draw per (seed, event)."""
    rows = conn.execute(
        _RANDOM_SQL,
        {"before": before, "ev_id": ev_id, "salt": f":{seed}:{ev_id}", "n": RANDOM_PASSAGES},
    ).fetchall()
    return [
        {
            "passage_id": r[0],
            "event_id": r[1],
            "event_date": r[2].isoformat(),
            "section_type": r[3],
            "content": r[4],
        }
        for r in rows
    ]


class _AgentSystem:
    """Shared plumbing for the systems built on ``causetrace.agent.Agent``."""

    name: str
    verify: bool

    def __init__(
        self, provider: Provider, agent_factory: Callable[[Connection, bool], Agent]
    ) -> None:
        self.provider = provider
        self.model = provider.model
        self.agent_factory = agent_factory
        self.prompts = load_prompts()
        self._agents: dict[int, tuple[Connection, Agent]] = {}  # one agent per worker connection

    def agent(self, conn: Connection) -> Agent:
        cached = self._agents.get(id(conn))
        if cached is None or cached[0] is not conn:
            cached = (conn, self.agent_factory(conn, self.verify))
            self._agents[id(conn)] = cached
        return cached[1]

    def config(self) -> dict[str, Any]:
        return {
            "prompts": {n: p.sha256 for n, p in self.prompts.items()},
            "limits": asdict(AGENT_LIMITS),
            "verify": self.verify,
        }


class RandomContext(_AgentSystem):
    """The agent's analyze step with 10 random history passages instead of retrieved ones."""

    name = "random_context"
    verify = False

    def __init__(
        self,
        provider: Provider,
        agent_factory: Callable[[Connection, bool], Agent],
        seed: int,
    ) -> None:
        super().__init__(provider, agent_factory)
        self.seed = seed

    def config(self) -> dict[str, Any]:
        return {**super().config(), "seed": self.seed, "passages": RANDOM_PASSAGES}

    def predict(self, conn: Connection, ev_id: str) -> dict[str, Any]:
        start = time.perf_counter()
        agent = self.agent(conn)
        packet = get_incident_packet(ev_id, conn)
        passages = random_passages(conn, ev_id, corpus_cutoff(packet.ev_date), self.seed)
        update = agent.analyze(
            {
                "packet": packet.model_dump(mode="json"),
                "passages": passages,
                "tokens_used": 0,
                "feedback": [],
            }
        )
        findings = update.get("predictions", [])
        labels = _labels(findings)
        return {
            "labels": labels,
            "abstained": not labels,
            "cites": True,
            "findings": findings,
            "citations": _citations(findings),
            "context_passages": [p["passage_id"] for p in passages],
            **_trace_usage(update.get("trace", [])),
            "seconds": time.perf_counter() - start,
        }


def _citations(findings: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    out = []
    for f in findings:
        for c in f.get("citations") or []:
            if isinstance(c, str):
                out.append({"label": str(f.get("label", "")).strip(), "passage_id": c})
    return out


class Rag(_AgentSystem):
    """The merged agent with verify skipped (``Agent(verify=False)``)."""

    name = "rag"
    verify = False

    def predict(self, conn: Connection, ev_id: str) -> dict[str, Any]:
        start = time.perf_counter()
        result = self.agent(conn).run(ev_id)
        findings = result["findings"]
        out = {
            "labels": _labels(findings),
            "abstained": bool(result["abstained"]),
            "cites": True,
            "findings": findings,
            "citations": _citations(findings),
            "context_passages": [p["passage_id"] for p in result["passages"]],
            "queries": result["queries"],
            "hops": result["hops"],
            "revisions": result["revisions"],
            **_trace_usage(result["trace"]),
            "seconds": time.perf_counter() - start,
        }
        if self.verify:
            findings_checked = (result.get("verification") or {}).get("findings", [])
            verdicts = [v for f in findings_checked for v in f.get("verdicts", {}).values()]
            out["verifier"] = {
                "pairs": len(verdicts),
                "supported": sum(v == "SUPPORTED" for v in verdicts),
            }
        return out


class RagVerify(Rag):
    """The full merged agent."""

    name = "rag_verify"
    verify = True
