"""Bounded multi-hop agent: plan -> retrieve -> check_evidence -> (one gap hop) ->
analyze -> verify -> (one revision) -> answer or abstain.

Model input for the target event is its packet only (``causetrace.packets``); retrieval
is history-only (``corpus_cutoff`` and ``exclude_event_id``). Every LLM call goes through
a ``causetrace.llm`` provider, which applies the MAX_USD guard. Loops are bounded by the
hop and revision counters in ``AgentLimits`` and by a token ceiling, so the graph always
terminates.
"""

import operator
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from causetrace.agent.prompts import Prompt, load_prompts
from causetrace.agent.taxonomy import load_taxonomy, render_taxonomy, taxonomy_sha256
from causetrace.config import AGENT_LIMITS, AgentLimits
from causetrace.db import Connection
from causetrace.evaluation.retrieval_eval import packet_query_text
from causetrace.llm import Effort, LLMResponseError, Provider
from causetrace.packets import IncidentPacket, get_incident_packet
from causetrace.retrieval import Mode, Retriever, rrf_fuse
from causetrace.split import corpus_cutoff
from causetrace.telemetry import tracer

NodeName = Literal["plan", "check_evidence", "analyze", "verify"]
EFFORT: dict[NodeName, Effort] = {
    "plan": "low",
    "check_evidence": "low",
    "analyze": "medium",
    "verify": "medium",
}
VERDICTS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED")
# Supersteps in the longest path are 9; the limit only guards against a routing bug.
RECURSION_LIMIT = 25

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}
CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "enough": {"type": "boolean"},
        "missing": {"type": "string"},
        "gap_query": {"type": "string"},
    },
    "required": ["enough", "missing", "gap_query"],
    "additionalProperties": False,
}
ANALYZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                },
                "required": ["label", "confidence", "citations", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}
VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "passage_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "reason": {"type": "string"},
                },
                "required": ["label", "passage_id", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


class AgentState(TypedDict, total=False):
    ev_id: str
    before_date: date
    packet: dict[str, Any]
    queries: list[str]
    gap_query: str | None
    ranked_lists: list[list[dict[str, Any]]]  # one ranked passage list per query searched
    passages: list[dict[str, Any]]
    evidence_ok: bool
    hops: int
    predictions: list[dict[str, Any]]
    feedback: list[str]
    verification: dict[str, Any]
    revisions: int
    answer: list[dict[str, Any]]
    abstained: bool
    tokens_used: Annotated[int, operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]


def describe_packet(packet: IncidentPacket) -> str:
    """The packet as prompt text: SAFE fields and the gated factual narratives."""
    loc, wx, inj = packet.location, packet.weather, packet.injuries
    lines = [
        f"Date: {packet.ev_date.isoformat()}",
        "Location: " + ", ".join(x for x in (loc.city, loc.state, loc.country) if x),
        f"Weather: {wx.basic_conditions or 'unknown'}, light {wx.light_condition or 'unknown'}, "
        f"wind {wx.wind_speed_kts if wx.wind_speed_kts is not None else '?'} kt, "
        f"visibility {wx.visibility_sm if wx.visibility_sm is not None else '?'} sm",
        f"Injuries: highest {inj.highest or 'unknown'}, fatal {inj.fatal}, serious {inj.serious}",
    ]
    for a in packet.aircraft:
        lines.append(
            f"Aircraft {a.aircraft_key}: {a.category or '?'} {a.make or ''} {a.model or ''}"
            f", damage {a.damage or 'unknown'}, phase {a.phase_of_flight or 'unknown'}"
        )
    narratives = [a.narr_accp for a in packet.aircraft if a.narr_accp]
    lines.append("")
    lines.append("Factual narrative:")
    lines.append("\n\n".join(narratives) if narratives else "(withheld)")
    return "\n".join(lines)


def render_passages(passages: list[dict[str, Any]]) -> str:
    if not passages:
        return "(none)"
    return "\n\n".join(
        f"[{p['passage_id']}] {p['section_type']} of event {p['event_id']} "
        f"({p['event_date']})\n{p['content']}"
        for p in passages
    )


def _clean_queries(raw: Any, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for q in raw:
        if isinstance(q, str) and q.strip() and q.strip() not in out:
            out.append(q.strip())
    return out[:limit]


class Agent:
    """Runs the graph for one target event at a time."""

    def __init__(
        self,
        conn: Connection,
        retriever: Retriever,
        provider: Provider,
        *,
        limits: AgentLimits = AGENT_LIMITS,
        verify: bool = True,
    ) -> None:
        """``verify=False`` skips the verify node: analyze's predictions are the answer."""
        self.conn = conn
        self.verify_enabled = verify
        self.retriever = retriever
        self.provider = provider
        self.limits = limits
        self.prompts: dict[str, Prompt] = load_prompts()
        self.taxonomy = load_taxonomy(conn)
        self.system_blocks = [self.prompts["system"].text, render_taxonomy(self.taxonomy)]
        self.graph = self._build()

    # --- LLM calls ------------------------------------------------------------------

    def _call(
        self, state: AgentState, node: NodeName, schema: dict[str, Any], **values: object
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """One provider call; returns (parsed or None, trace entry with usage)."""
        entry: dict[str, Any] = {"node": node, "prompt_sha256": self.prompts[node].sha256}
        if state.get("tokens_used", 0) >= self.limits.max_tokens:
            entry["skipped"] = "token ceiling reached"
            return None, entry
        prompt = self.prompts[node].render(**values)
        try:
            parsed, usage = self.provider.generate_json(
                self.system_blocks, prompt, schema, effort=EFFORT[node], cache_system=True
            )
        except LLMResponseError as err:
            entry["error"] = str(err)
            entry["tokens"] = 0
            return None, entry
        entry["model"] = usage.model
        entry["effort"] = EFFORT[node]
        entry["tokens"] = (
            usage.input_tokens
            + usage.output_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens
        )
        entry["cost_usd"] = usage.cost_usd
        entry["latency_s"] = round(usage.latency_s, 3)
        entry["cache_read_tokens"] = usage.cache_read_input_tokens
        return parsed, entry

    def _packet(self, state: AgentState) -> IncidentPacket:
        return IncidentPacket.model_validate(state["packet"])

    # --- Nodes ----------------------------------------------------------------------

    def plan(self, state: AgentState) -> dict[str, Any]:
        packet = self._packet(state)
        values = {
            "packet": describe_packet(packet),
            "max_queries": self.limits.max_queries,
            "max_query_words": self.limits.rerank_max_words,
        }
        trace: list[dict[str, Any]] = []
        queries: list[str] = []
        for attempt in (1, 2):  # retry once on invalid JSON
            parsed, entry = self._call(state, "plan", PLAN_SCHEMA, **values)
            entry["attempt"] = attempt
            trace.append(entry)
            if parsed is not None:
                queries = _clean_queries(parsed.get("queries"), self.limits.max_queries)
                if not queries:
                    entry["error"] = "no usable queries"
            if queries or "error" not in entry:
                break
        fallback = not queries
        if fallback:
            queries = [packet_query_text(packet)]
        trace.append({"node": "plan", "queries": queries, "fallback_to_packet": fallback})
        tokens = sum(int(t.get("tokens", 0)) for t in trace)
        return {"queries": queries, "trace": trace, "tokens_used": tokens}

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        gap = state.get("gap_query")
        hop = bool(gap)  # check_evidence sets gap_query only when an extra hop is allowed
        new_queries = [gap] if gap else state["queries"]
        ranked_lists = list(state.get("ranked_lists", []))
        searches = []
        for query in new_queries:
            mode: Mode = (
                "rrf_rerank" if len(query.split()) < self.limits.rerank_max_words else "rrf"
            )
            start = time.perf_counter()
            result = self.retriever.search(
                query,
                self.limits.per_query_top_k,
                before_date=state["before_date"],
                exclude_event_id=state["ev_id"],
                mode=mode,
            )
            ranked_lists.append(
                [
                    {
                        "passage_id": h.passage_id,
                        "event_id": h.event_id,
                        "event_date": h.metadata["event_date"].isoformat(),
                        "section_type": h.metadata["section_type"],
                        "content": h.content,
                    }
                    for h in result.hits
                ]
            )
            search: dict[str, Any] = {
                "query": query,
                "mode": mode,
                "hits": len(result.hits),
                "ms": round((time.perf_counter() - start) * 1000, 1),
            }
            if result.fallback:
                search["fallback"] = result.fallback
            searches.append(search)
        by_id = {p["passage_id"]: p for ranked in ranked_lists for p in ranked}
        fused = rrf_fuse(
            [[p["passage_id"] for p in ranked] for ranked in ranked_lists],
            limit=self.limits.max_passages,
        )
        passages = [by_id[pid] for pid, _ in fused]
        for p in passages:  # retrieval enforces this in SQL; fail loudly if it ever breaks
            if (
                p["event_id"] == state["ev_id"]
                or p["event_date"] >= state["before_date"].isoformat()
            ):
                raise AssertionError(f"leakage: retrieved {p['passage_id']}")
        update: dict[str, Any] = {
            "ranked_lists": ranked_lists,
            "passages": passages,
            "trace": [
                {
                    "node": "retrieve",
                    "hop": hop,
                    "searches": searches,
                    "passages": [p["passage_id"] for p in passages],
                }
            ],
        }
        if hop:
            update["hops"] = state.get("hops", 0) + 1
            update["gap_query"] = None
        return update

    def check_evidence(self, state: AgentState) -> dict[str, Any]:
        parsed, entry = self._call(
            state,
            "check_evidence",
            CHECK_SCHEMA,
            packet=describe_packet(self._packet(state)),
            passages=render_passages(state.get("passages", [])),
            max_query_words=self.limits.rerank_max_words,
        )
        enough = True if parsed is None else bool(parsed.get("enough", True))
        gap = None if parsed is None else str(parsed.get("gap_query", "")).strip() or None
        entry.update({"enough": enough, "missing": (parsed or {}).get("missing"), "gap_query": gap})
        update: dict[str, Any] = {
            "evidence_ok": enough,
            "trace": [entry],
            "tokens_used": int(entry.get("tokens", 0)),
        }
        if not enough and gap and state.get("hops", 0) < self.limits.max_extra_hops:
            update["gap_query"] = gap
        return update

    def analyze(self, state: AgentState) -> dict[str, Any]:
        feedback = state.get("feedback") or []
        revision = bool(feedback)  # only verify leaves feedback
        parsed, entry = self._call(
            state,
            "analyze",
            ANALYZE_SCHEMA,
            packet=describe_packet(self._packet(state)),
            passages=render_passages(state.get("passages", [])),
            max_findings=self.limits.max_findings,
            feedback="\n".join(f"- {f}" for f in feedback) if feedback else "(none)",
        )
        raw = (parsed or {}).get("findings", [])
        findings = [f for f in raw if isinstance(f, dict)] if isinstance(raw, list) else []
        findings.sort(key=lambda f: -_confidence(f))
        entry.update(
            {
                "revision": revision,
                "returned": len(findings),
                "kept": min(len(findings), self.limits.max_findings),
            }
        )
        update: dict[str, Any] = {
            "predictions": findings[: self.limits.max_findings],
            "feedback": [],
            "trace": [entry],
            "tokens_used": int(entry.get("tokens", 0)),
        }
        if revision:
            update["revisions"] = state.get("revisions", 0) + 1
        return update

    def verify(self, state: AgentState) -> dict[str, Any]:
        passages = {p["passage_id"]: p for p in state.get("passages", [])}
        cutoff = state["before_date"].isoformat()
        incident_date = self._packet(state).ev_date.isoformat()
        checked: list[dict[str, Any]] = []
        feedback: list[str] = []
        predictions = state.get("predictions", [])
        if len(predictions) > self.limits.max_findings:
            feedback.append(f"Return at most {self.limits.max_findings} findings.")
        for f in predictions[: self.limits.max_findings]:
            label = str(f.get("label", "")).strip()
            problems: list[str] = []  # fatal: the finding is rejected
            notes: list[str] = []  # non-fatal: the citation is dropped
            if label not in self.taxonomy:
                problems.append(f"label {label!r} is not in the taxonomy")
            confidence = f.get("confidence")
            if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
                problems.append("confidence must be a number from 0 to 1")
            cites = [str(c) for c in f.get("citations", []) if isinstance(c, str)]
            valid: list[str] = []
            for cid in dict.fromkeys(cites):
                p = passages.get(cid)
                if p is None:
                    notes.append(f"citation {cid} was not retrieved")
                elif p["event_date"] >= cutoff or p["event_date"] >= incident_date:
                    notes.append(f"citation {cid} does not predate the incident")
                else:
                    valid.append(cid)
            if not valid:
                problems.append("no valid citation")
            checked.append(
                {"label": label, "confidence": confidence, "citations": valid,
                 "rationale": f.get("rationale"), "problems": problems, "notes": notes,
                 "verdicts": {}}
            )  # fmt: skip
            feedback += [f"{label}: {msg}" for msg in problems + notes]

        to_check = [c for c in checked if c["citations"] and not c["problems"]]
        entry: dict[str, Any] = {"node": "verify", "deterministic_problems": len(feedback)}
        tokens = 0
        if to_check:
            pairs = "\n\n".join(
                f"Finding {c['label']} ({self.taxonomy[c['label']]}): {c['rationale']}\n"
                + "\n".join(
                    f"  - passage [{pid}]: {passages[pid]['content']}" for pid in c["citations"]
                )
                for c in to_check
            )
            parsed, entry = self._call(
                state,
                "verify",
                VERIFY_SCHEMA,
                packet=describe_packet(self._packet(state)),
                pairs=pairs,
            )
            entry["deterministic_problems"] = len(feedback)
            tokens = int(entry.get("tokens", 0))
            for v in (parsed or {}).get("verdicts", []) if parsed else []:
                if not isinstance(v, dict) or v.get("verdict") not in VERDICTS:
                    continue
                for c in to_check:
                    if c["label"] == v.get("label") and v.get("passage_id") in c["citations"]:
                        c["verdicts"][v["passage_id"]] = v["verdict"]
        for c in checked:
            supported = [pid for pid in c["citations"] if c["verdicts"].get(pid) == "SUPPORTED"]
            c["supported_citations"] = supported
            c["status"] = "supported" if supported and not c["problems"] else "rejected"
            if c["citations"] and not supported and not c["problems"]:
                verdicts = ", ".join(
                    f"{pid}: {c['verdicts'].get(pid, 'UNCHECKED')}" for pid in c["citations"]
                )
                feedback.append(f"{c['label']}: no citation was judged SUPPORTED ({verdicts})")
        needs_revision = bool(feedback) or not any(c["status"] == "supported" for c in checked)
        entry["supported"] = sum(c["status"] == "supported" for c in checked)
        entry["needs_revision"] = needs_revision
        return {
            "verification": {"findings": checked, "needs_revision": needs_revision},
            "feedback": feedback or (["No finding was supported."] if needs_revision else []),
            "trace": [entry],
            "tokens_used": tokens,
        }

    def finalize(self, state: AgentState) -> dict[str, Any]:
        if not self.verify_enabled:
            answer = [
                {
                    "label": str(f.get("label", "")).strip(),
                    "short_name": self.taxonomy.get(str(f.get("label", "")).strip()),
                    "confidence": f.get("confidence"),
                    "citations": [c for c in f.get("citations", []) if isinstance(c, str)],
                    "rationale": f.get("rationale"),
                }
                for f in state.get("predictions", [])
            ][: self.limits.max_findings]
            return {
                "answer": answer,
                "abstained": not answer,
                "trace": [{"node": "answer" if answer else "abstain", "verify": False}],
            }
        findings = (state.get("verification") or {}).get("findings", [])
        answer = [
            {
                "label": c["label"],
                "short_name": self.taxonomy.get(c["label"]),
                "confidence": c["confidence"],
                "citations": c["supported_citations"],
                "rationale": c["rationale"],
            }
            for c in findings
            if c.get("status") == "supported"
        ][: self.limits.max_findings]
        reason = None
        if not answer:
            reason = "no passages" if not state.get("passages") else "no supported finding"
        return {
            "answer": answer,
            "abstained": not answer,
            "trace": [{"node": "answer" if answer else "abstain", "reason": reason}],
        }

    # --- Routing --------------------------------------------------------------------

    def _out_of_tokens(self, state: AgentState) -> bool:
        return state.get("tokens_used", 0) >= self.limits.max_tokens

    def after_retrieve(self, state: AgentState) -> str:
        if not state.get("passages"):
            return "finalize"
        if state.get("hops", 0) == 0 and not self._out_of_tokens(state):
            return "check_evidence"
        return "analyze"

    def after_check(self, state: AgentState) -> str:
        if state.get("gap_query") and state.get("hops", 0) < self.limits.max_extra_hops:
            return "retrieve"
        return "analyze"

    def after_verify(self, state: AgentState) -> str:
        needs = (state.get("verification") or {}).get("needs_revision", False)
        if (
            needs
            and state.get("revisions", 0) < self.limits.max_revisions
            and not self._out_of_tokens(state)
        ):
            return "analyze"
        return "finalize"

    @staticmethod
    def _traced(name: str, node: Callable[[AgentState], dict[str, Any]]) -> Any:
        """Run ``node`` in an ``agent.<name>`` span (no change to what it does)."""

        def traced(state: AgentState) -> dict[str, Any]:
            with tracer.start_as_current_span(f"agent.{name}"):
                return node(state)

        return traced

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("plan", self._traced("plan", self.plan))
        g.add_node("retrieve", self._traced("retrieve", self.retrieve))
        g.add_node("check_evidence", self._traced("check_evidence", self.check_evidence))
        g.add_node("analyze", self._traced("analyze", self.analyze))
        if self.verify_enabled:
            g.add_node("verify", self._traced("verify", self.verify))
        g.add_node("finalize", self._traced("finalize", self.finalize))
        g.add_edge(START, "plan")
        g.add_edge("plan", "retrieve")
        g.add_conditional_edges(
            "retrieve", self.after_retrieve, ["check_evidence", "analyze", "finalize"]
        )
        g.add_conditional_edges("check_evidence", self.after_check, ["retrieve", "analyze"])
        if self.verify_enabled:
            g.add_edge("analyze", "verify")
            g.add_conditional_edges("verify", self.after_verify, ["analyze", "finalize"])
        else:
            g.add_edge("analyze", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    # --- Entry point ----------------------------------------------------------------

    def run(self, ev_id: str) -> dict[str, Any]:
        """Run the agent for ``ev_id``; returns the result with its full trace."""
        return self.run_packet(get_incident_packet(ev_id, self.conn))

    def run_packet(self, packet: IncidentPacket) -> dict[str, Any]:
        """Run the agent on a packet (from ``get_incident_packet`` or a free-text request).

        Retrieval uses ``corpus_cutoff(packet.ev_date)`` and excludes ``packet.ev_id``.
        """
        with tracer.start_as_current_span("agent.workflow") as span:
            result = self._run(packet)
            span.set_attribute("workflow.ev_id", result["ev_id"])
            span.set_attribute("workflow.hops", result["hops"])
            span.set_attribute("workflow.revisions", result["revisions"])
            span.set_attribute("workflow.abstained", bool(result["abstained"]))
            span.set_attribute("workflow.findings", len(result["findings"]))
            span.set_attribute("workflow.passages", len(result["passages"]))
            span.set_attribute("workflow.llm_calls", result["usage"]["calls"])
            span.set_attribute("workflow.tokens", result["usage"]["tokens"])
        return result

    def _run(self, packet: IncidentPacket) -> dict[str, Any]:
        ev_id = packet.ev_id
        before = corpus_cutoff(packet.ev_date)
        initial: AgentState = {
            "ev_id": ev_id,
            "before_date": before,
            "packet": packet.model_dump(mode="json"),
            "queries": [],
            "gap_query": None,
            "ranked_lists": [],
            "passages": [],
            "evidence_ok": False,
            "hops": 0,
            "revisions": 0,
            "abstained": False,
            "answer": [],
            "tokens_used": 0,
            "trace": [],
        }
        start = time.perf_counter()
        calls_before = len(self.provider.tracker.calls)
        spent_before = self.provider.tracker.spent_usd
        try:
            state = self.graph.invoke(initial, {"recursion_limit": RECURSION_LIMIT})
        except GraphRecursionError:  # unreachable with the bounded routes; abstain if it happens
            state = {**initial, "abstained": True,
                     "trace": [{"node": "abstain", "reason": "recursion limit"}]}  # fmt: skip
        return {
            "ev_id": ev_id,
            "before_date": before.isoformat(),
            "abstained": state.get("abstained", True),
            "findings": state.get("answer", []),
            "queries": state.get("queries", []),
            "passages": [
                {k: p[k] for k in ("passage_id", "event_id", "event_date", "section_type")}
                for p in state.get("passages", [])
            ],
            "evidence_ok": state.get("evidence_ok"),
            "hops": state.get("hops", 0),
            "revisions": state.get("revisions", 0),
            "predictions": state.get("predictions", []),
            "verification": state.get("verification", {}),
            "usage": {
                "model": self.provider.model,
                "calls": len(self.provider.tracker.calls) - calls_before,
                "tokens": state.get("tokens_used", 0),
                "cost_usd": round(self.provider.tracker.spent_usd - spent_before, 6),
                "seconds": round(time.perf_counter() - start, 2),
            },
            "prompts": {name: p.sha256 for name, p in self.prompts.items()},
            "taxonomy": {"codes": len(self.taxonomy), "sha256": taxonomy_sha256(self.taxonomy)},
            "limits": asdict(self.limits),
            "trace": state.get("trace", []),
        }


def _confidence(f: dict[str, Any]) -> float:
    c = f.get("confidence")
    return float(c) if isinstance(c, int | float) else 0.0
