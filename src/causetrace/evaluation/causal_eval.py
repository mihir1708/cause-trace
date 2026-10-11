"""Compare the six causal systems on held-out incidents (docs/EVAL_PROTOCOL.md).

Resumable: every case result is appended to ``artifacts/runs/<system>__<model>.jsonl``
under a key of (case, repeat, config hash); a rerun skips what is cached. LLM systems use
``AnthropicProvider`` directly (no fallback), so one run never mixes models; when Claude is
unavailable or MAX_USD is reached the run stops and can be resumed.

``--split dev`` runs the LLM systems on 2023 history events instead (the tuning set) and
writes ``artifacts/causal_dev_results.*``; their results tune each LLM system's confidence
cut-off, which the held-out report then applies to cached results without new calls.

Usage: python scripts/eval_causal.py [--split dev] [--cases N] [--systems ...] [--limit N]
       [--model M] [--repeat R] [--workers N]
"""

import argparse
import json
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

from causetrace.agent import Agent
from causetrace.agent.taxonomy import load_taxonomy
from causetrace.config import LLM_MODEL, max_usd
from causetrace.db import Connection, connect
from causetrace.embeddings import Embedder, default_embedder
from causetrace.evaluation.causal_metrics import (
    PassageInfo,
    aggregate_repeats,
    gold_outside,
    score_cases,
)
from causetrace.evaluation.causal_systems import (
    FREE_SYSTEMS,
    KNN_K,
    KNN_MODE,
    LLM_SYSTEMS,
    SYSTEMS,
    ClosedBook,
    FrequencyPrior,
    KnnVote,
    Rag,
    RagVerify,
    RandomContext,
    System,
    apply_cutoff,
    knn_neighbours,
    knn_predict,
    micro_f1,
    sha256_json,
    tune_confidence,
    tune_tau,
)
from causetrace.evaluation.citation_audit import (
    DEFAULT_AUDIT,
    audit_summary,
    write_audit_sample,
)
from causetrace.evaluation.retrieval_eval import git_sha, model_names, scored_held_out_events
from causetrace.gold import get_all_gold_labels
from causetrace.llm import BudgetExceededError, Provider, ProviderUnavailableError, SpendTracker
from causetrace.rerank import Reranker, default_reranker
from causetrace.retrieval import Retriever
from causetrace.split import HELD_OUT_FROM, corpus_cutoff

DEFAULT_CASES = Path("evals/causal_cases.json")
DEFAULT_DEV_CASES = Path("evals/causal_dev_cases.json")
DEFAULT_RUNS = Path("artifacts/runs")
DEFAULT_JSON = Path("artifacts/causal_results.json")
DEFAULT_MD = Path("artifacts/causal_results.md")
DEFAULT_DEV_JSON = Path("artifacts/causal_dev_results.json")
DEFAULT_DEV_MD = Path("artifacts/causal_dev_results.md")
DEV_RUNS = "dev"  # dev results live in <runs-dir>/dev, apart from held-out results
HELD_OUT_DESCRIPTION = (
    "Held-out events with gold labels and narr_accp (docs/EVAL_PROTOCOL.md); "
    "frozen for the causal comparison."
)
DEV_DESCRIPTION = (
    "2023 history events with gold labels and narr_accp (the tuning set, "
    "docs/EVAL_PROTOCOL.md); frozen for tuning the LLM systems' confidence cut-offs."
)
TUNING_FROM = date(2023, 1, 1)
DEFAULT_SEED = 42

ProviderFactory = Callable[[str, SpendTracker], Provider]
RetrieverFactory = Callable[[Connection], Retriever]


# --- Cases ----------------------------------------------------------------------------


def freeze_cases(
    path: Path,
    scored: Sequence[str],
    n: int | None,
    seed: int,
    refreeze: bool = False,
    description: str = HELD_OUT_DESCRIPTION,
) -> list[str]:
    """The frozen case list: reuse ``path`` when it matches the request, else write it."""
    if path.is_file() and not refreeze:
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if n is None or frozen.get("n_requested") == n:
            missing = set(frozen["ev_ids"]) - set(scored)
            if missing:
                raise SystemExit(f"{path}: {len(missing)} frozen cases are not scored events")
            return list(frozen["ev_ids"])
        raise SystemExit(
            f"{path} holds {frozen.get('n_requested')} cases, --cases asks for {n}; "
            "pass --refreeze to replace it"
        )
    ev_ids = (
        sorted(scored)
        if n is None or n >= len(scored)
        else sorted(random.Random(seed).sample(sorted(scored), n))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "description": description,
                "seed": seed,
                "n_requested": n,
                "n_scored": len(scored),
                "ev_ids": ev_ids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ev_ids


_TUNING_SQL = """
SELECT e.ev_id FROM events e
WHERE e.ev_date >= %s AND e.ev_date < %s
  AND EXISTS (SELECT 1 FROM findings f WHERE f.ev_id = e.ev_id AND f.cm_inpc IS TRUE)
  AND EXISTS (SELECT 1 FROM narratives n
              WHERE n.ev_id = e.ev_id AND coalesce(btrim(n.narr_accp), '') <> '')
ORDER BY e.ev_id
"""


def tuning_events(conn: Connection, limit: int | None, seed: int) -> list[str]:
    """2023 history events with gold labels and narr_accp (the tuning set)."""
    events = [r[0] for r in conn.execute(_TUNING_SQL, (TUNING_FROM, HELD_OUT_FROM))]
    if limit is not None and limit < len(events):
        events = sorted(random.Random(seed).sample(events, limit))
    return events


# --- Cache ----------------------------------------------------------------------------


class ResultCache:
    """Append-only JSONL per (system, model); thread-safe appends."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def path(self, system: str, model: str) -> Path:
        return self.root / f"{system}__{model.replace('/', '_')}.jsonl"

    def load(self, system: str, model: str) -> dict[tuple[str, int, str], dict[str, Any]]:
        path = self.path(system, model)
        records: dict[tuple[str, int, str], dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    records[(r["ev_id"], int(r["repeat"]), r["config_hash"])] = r
        return records

    def models(self, system: str) -> list[str]:
        prefix = f"{system}__"
        return sorted(
            p.stem[len(prefix) :] for p in self.root.glob(f"{prefix}*.jsonl") if p.is_file()
        )

    def append(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.path(record["system"], record["model"]).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                f.flush()


class NeighbourCache:
    """kNN neighbour lists per event, keyed by the retrieval configuration."""

    def __init__(self, root: Path, retrieval_config: Mapping[str, Any]) -> None:
        self.path = root / "knn_neighbours.jsonl"
        self.config_hash = sha256_json(dict(retrieval_config))
        self._lock = threading.Lock()
        self.lists: dict[str, list[str]] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["config_hash"] == self.config_hash:
                        self.lists[r["ev_id"]] = r["neighbours"]

    def get(self, conn: Connection, ev_id: str, compute: RetrieverFactory) -> list[str]:
        if ev_id in self.lists:
            return self.lists[ev_id]
        neighbours = knn_neighbours(compute(conn), conn, ev_id)
        with self._lock:
            self.lists[ev_id] = neighbours
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"ev_id": ev_id, "config_hash": self.config_hash, "neighbours": neighbours}
                    )
                    + "\n"
                )
        return neighbours


# --- Worker resources -----------------------------------------------------------------


class LockedEmbedder:
    """Serializes calls to a shared embedding model across worker threads."""

    def __init__(self, inner: Embedder, lock: threading.Lock) -> None:
        self.inner = inner
        self.model_name, self.revision, self.dim = inner.model_name, inner.revision, inner.dim
        self._lock = lock

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        with self._lock:
            return self.inner.embed(texts, is_query=is_query)


class LockedReranker:
    def __init__(self, inner: Reranker, lock: threading.Lock) -> None:
        self.inner = inner
        self._lock = lock

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        with self._lock:
            return self.inner.score(query, passages)


@dataclass
class Workers:
    """One connection and retriever per worker thread."""

    db_url: str | None
    retriever_factory: RetrieverFactory
    _local: threading.local = field(default_factory=threading.local)
    _conns: list[Connection] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def conn(self) -> Connection:
        conn: Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.db_url)
            conn.autocommit = True
            self._local.conn = conn
            with self._lock:
                self._conns.append(conn)
        return conn

    def retriever(self, conn: Connection) -> Retriever:
        cached = getattr(self._local, "retriever", None)
        if cached is None or cached.conn is not conn:
            cached = self.retriever_factory(conn)
            self._local.retriever = cached
        return cached

    def close(self) -> None:
        for conn in self._conns:
            conn.close()


def default_retriever_factory() -> RetrieverFactory:
    embedder: Embedder | None = None
    reranker: Reranker | None = None
    lock = threading.Lock()
    # One lock for both models: they share the accelerator, and PyTorch's MPS (Apple GPU)
    # backend aborts when two threads encode GPU work at the same time.
    model_lock = threading.Lock()

    def make(conn: Connection) -> Retriever:
        nonlocal embedder, reranker
        with lock:
            if embedder is None:
                embedder = LockedEmbedder(default_embedder(), model_lock)
                reranker = LockedReranker(default_reranker(), model_lock)
        return Retriever(conn, embedder, reranker)

    return make


# --- Running --------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    system: System
    ev_id: str
    repeat: int


def run_tasks(
    tasks: Sequence[Task],
    workers: Workers,
    cache: ResultCache,
    tracker: SpendTracker,
    n_workers: int,
    log: Callable[[str], None] = print,
) -> str | None:
    """Run tasks in parallel; return a stop reason, or None when all finished."""
    stop: list[str] = []
    stop_event = threading.Event()

    def one(task: Task) -> None:
        if stop_event.is_set():
            return
        conn = workers.conn()
        start = time.perf_counter()
        try:
            result = task.system.predict(conn, task.ev_id)
        except (BudgetExceededError, ProviderUnavailableError) as err:
            stop_event.set()
            stop.append(f"{type(err).__name__}: {err}")
            return
        result.setdefault("seconds", time.perf_counter() - start)
        record = {
            "ev_id": task.ev_id,
            "system": task.system.name,
            "model": task.system.model,
            "repeat": task.repeat,
            "config_hash": sha256_json(task.system.config()),
            **result,
        }
        cache.append(record)
        if task.system.model != "none":
            log(
                f"{task.system.name} [{task.system.model}] {task.ev_id} r{task.repeat}: "
                f"${float(result.get('cost_usd', 0.0)):.4f}, {result['seconds']:.1f}s "
                f"(run total ${tracker.spent_usd:.4f})"
            )

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        futures: list[Future[None]] = [pool.submit(one, t) for t in tasks]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for f in done:
            f.result()  # re-raise unexpected errors
    return stop[0] if stop else None


# --- Report ---------------------------------------------------------------------------


def _fmt(row: Mapping[str, Any], key: str, digits: int = 3) -> str:
    value = row.get(key)
    if value is None:
        return "-"
    text = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    lo, hi = row.get(f"{key}_min"), row.get(f"{key}_max")
    if isinstance(lo, int | float) and isinstance(hi, int | float) and hi != lo:
        text += f" ±{(hi - lo) / 2:.{digits}f}"
    return text


def render_markdown(report: Mapping[str, Any]) -> str:
    out = [
        "# Causal system comparison",
        "",
        "Generated by `scripts/eval_causal.py`; do not edit by hand.",
        "",
        f"- Git SHA: `{report['git_sha']}`",
        f"- Cases: {report['cases']['n']} of {report['cases']['n_scored']} scored held-out "
        f"events (`{report['cases']['path']}`, seed {report['cases']['seed']})",
        f"- Taxonomy: {report['taxonomy']['codes']} codes; gold labels outside it: "
        f"{report['taxonomy']['gold_outside_cases']['labels']} in "
        f"{report['taxonomy']['gold_outside_cases']['cases']} cases "
        f"({report['taxonomy']['gold_outside_all']['labels']} over all scored events). "
        "They count as misses for every system.",
        f"- frequency_prior: k = {report['tuning']['frequency_prior_k']}; knn_vote: cut-off "
        f"{report['tuning']['knn_tau']} (tuned on {report['tuning']['n_events']} 2023 events, "
        f"micro F1); retrieval `{report['retrieval']}`",
        *_cutoff_lines(report["tuning"]),
        f"- LLM cost in this report: ${report['total_cost_usd']:.4f}",
        "",
        "Abstentions count as empty predictions in P/R/F1; *F1 answered* uses answered cases "
        "only. ± is half the range over repeats. *Cut-off*: LLM findings with a lower "
        "confidence are dropped (tuned on the dev split).",
        "",
        *_table(report["rows"]),
    ]
    audit = report.get("citation_audit")
    out += ["", "## Citation audit (rag_verify)", ""]
    if audit and audit["graded"]:
        out.append(
            f"{audit['graded']} of {audit['sampled']} sampled citations graded "
            "(grader: see the EVAL_PROTOCOL.md change log): "
            f"{_pct(audit['precision_strict'])} supports the label (grade 2), "
            f"{_pct(audit['precision_lenient'])} supports it at least partly (grade 1 or 2). "
            f"Verifier support rate for rag_verify: {_pct(audit['verifier_support_rate'])}."
        )
    elif audit:
        out.append(
            f"{audit['sampled']} citations sampled in `{audit['path']}`; none graded yet "
            "(`python scripts/grade_citations.py`)."
        )
    else:
        out.append("No rag_verify answers to sample yet.")
    if report.get("stopped"):
        out += ["", f"**Run stopped:** {report['stopped']} (resume by running again)."]
    return "\n".join(out) + "\n"


def _cutoff_lines(tuning: Mapping[str, Any]) -> list[str]:
    """One line per LLM system and model: its tuned cut-off and its dev scores."""
    dev = tuning["dev_cases"]
    out = []
    for key, c in sorted(tuning["llm_cutoffs"].items()):
        if c["tau"] is None:
            out.append(
                f"- {key}: no cut-off (dev run incomplete: {c['dev_done']} of "
                f"{c['dev_cases']} dev cases)"
            )
        else:
            out.append(
                f"- {key}: cut-off {c['tau']} (dev micro F1 {c['dev_micro_f1']:.3f}, "
                f"{c['dev_micro_f1_raw']:.3f} without; {c['dev_cases']} dev cases, "
                f"`{dev['path']}`)"
            )
    if not out:
        out.append("- LLM systems: no cut-off (no dev results; run `--split dev` first)")
    return out


def render_dev_markdown(report: Mapping[str, Any]) -> str:
    tuning = report["tuning"]
    out = [
        "# Causal systems on the dev split (2023 history events)",
        "",
        "Generated by `scripts/eval_causal.py --split dev`; do not edit by hand. These are "
        "tuning results, not held-out results.",
        "",
        f"- Git SHA: `{report['git_sha']}`",
        f"- Cases: {report['cases']['n']} of {report['cases']['n_scored']} 2023 events with "
        f"gold labels and narr_accp (`{report['cases']['path']}`, seed {report['cases']['seed']})"
        "; each searched before its own date, itself excluded",
        f"- Taxonomy: {report['taxonomy']['codes']} codes; gold labels outside it: "
        f"{report['taxonomy']['gold_outside_cases']['labels']} in "
        f"{report['taxonomy']['gold_outside_cases']['cases']} cases",
        *_cutoff_lines(tuning),
        f"- LLM cost in this report: ${report['total_cost_usd']:.4f}",
        "",
        "LLM systems are shown with their tuned cut-off and without one (cut-off `-`). "
        "frequency_prior and knn_vote were tuned on these events and are shown for reference.",
        "",
        *_table(report["rows"]),
    ]
    grids = [(k, c["scores"]) for k, c in sorted(tuning["llm_cutoffs"].items()) if c["scores"]]
    if grids:
        out += ["", "## Dev micro F1 by cut-off", ""]
        taus = sorted(grids[0][1])
        out.append("| System | " + " | ".join(str(t) for t in taus) + " |")
        out.append("|" + "---|" * (len(taus) + 1))
        for key, scores in grids:
            out.append(f"| {key} | " + " | ".join(f"{scores[t]:.3f}" for t in taus) + " |")
    if report.get("stopped"):
        out += ["", f"**Run stopped:** {report['stopped']} (resume by running again)."]
    return "\n".join(out) + "\n"


def _table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    out = [
        "| System | Model | Cut-off | Cases | Micro P | Micro R | Micro F1 | F1 answered "
        "| Macro F1 | Recovered | Abstain | Cit. valid | Cit. support | Verifier | Tokens "
        "| Calls | s | $ |",
        "|" + "---|" * 18,
    ]
    for row in rows:
        scope = row["n_cases"] if row["complete"] else f"{row['n_cases']} of {row['n_expected']}"
        if row.get("scope") == "all":
            scope = f"{scope} (all)"
        elif str(row.get("scope", "")).startswith("dev ("):
            scope = f"{scope} ({row['scope'][5:-1]})"
        out.append(
            f"| {row['system']} | {row['model']} | {_fmt(row, 'cutoff', 2)} | {scope} "
            f"| {_fmt(row, 'micro_precision')} "
            f"| {_fmt(row, 'micro_recall')} | {_fmt(row, 'micro_f1')} "
            f"| {_fmt(row, 'micro_f1_answered')} | {_fmt(row, 'macro_f1')} "
            f"| {_fmt(row, 'recovered', 0)} of {row['gold_findings']} "
            f"| {_fmt(row, 'abstention_rate')} | {_fmt(row, 'citation_validity')} "
            f"| {_fmt(row, 'citation_support')} | {_fmt(row, 'verifier_support_rate')} "
            f"| {_fmt(row, 'mean_tokens', 0)} | {_fmt(row, 'mean_calls', 1)} "
            f"| {_fmt(row, 'mean_seconds', 1)} | {_fmt(row, 'mean_cost_usd', 4)} |"
        )
    return out


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


# --- CLI ------------------------------------------------------------------------------


def _passage_info(conn: Connection, ids: set[str]) -> dict[str, PassageInfo]:
    if not ids:
        return {}
    rows = conn.execute(
        "SELECT passage_id, event_id, event_date FROM passages WHERE passage_id = ANY(%s)",
        (sorted(ids),),
    ).fetchall()
    return {r[0]: PassageInfo(r[1], r[2]) for r in rows}


def build_rows(
    conn: Connection,
    systems: Mapping[str, System],
    cache: ResultCache,
    cases: Sequence[str],
    scored: Sequence[str],
    gold: Mapping[str, frozenset[str]],
    history_gold: Mapping[str, frozenset[str]],
    cutoffs: Mapping[str, date],
    repeats: int,
    taus: Mapping[str, float | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], float]:
    """Score every (system, model) with cached results under the current configuration.

    ``taus`` maps ``"<system>__<model>"`` to the confidence cut-off applied to that LLM
    system's cached results before scoring (missing or None: no cut-off).

    Returns the rows, rag_verify answers (repeat 0, after the cut-off) per model, and the
    total LLM cost of the case results in the report.
    """
    rows: list[dict[str, Any]] = []
    answers: dict[str, list[dict[str, Any]]] = {}
    total_cost = 0.0
    for name in SYSTEMS:
        system = systems.get(name)
        if system is None:
            continue
        config_hash = sha256_json(system.config())
        models = [system.model] if name in FREE_SYSTEMS else cache.models(name)
        for model in models:
            records = [
                r for r in cache.load(name, model).values() if r["config_hash"] == config_hash
            ]
            tau = (taus or {}).get(cutoff_key(name, model)) if name in LLM_SYSTEMS else None
            records = [apply_cutoff(r, tau) for r in records]
            if name not in FREE_SYSTEMS:
                in_cases = set(cases)
                total_cost += sum(
                    float(r.get("cost_usd", 0.0))
                    for r in records
                    if r["ev_id"] in in_cases and r["repeat"] < repeats
                )
            scopes = (
                [("all", list(scored)), ("cases", list(cases))]
                if name in FREE_SYSTEMS
                else [("cases", list(cases))]
            )
            if name in FREE_SYSTEMS and set(cases) == set(scored):
                scopes = [("all", list(scored))]
            for scope, ev_ids in scopes:
                wanted = set(ev_ids)
                per_repeat = []
                n_rep = 1 if name in FREE_SYSTEMS else repeats
                for rep in range(n_rep):
                    res = [r for r in records if r["repeat"] == rep and r["ev_id"] in wanted]
                    if not res:
                        continue
                    pids = {c["passage_id"] for r in res for c in r.get("citations") or []}
                    metrics = score_cases(
                        res,
                        gold,
                        history_gold=history_gold,
                        passages=_passage_info(conn, pids),
                        cutoffs=cutoffs,
                    )
                    per_repeat.append(metrics)
                    if name == "rag_verify" and rep == 0:
                        answers[model] = res
                if not per_repeat:
                    continue
                row = aggregate_repeats(per_repeat)
                n_done = min(m["n_cases"] for m in per_repeat)
                row.update(
                    system=name,
                    model=model,
                    cutoff=tau,
                    scope=scope,
                    n_expected=len(ev_ids),
                    complete=n_done == len(ev_ids) and len(per_repeat) == n_rep,
                )
                rows.append(row)
    return rows, answers, total_cost


def cutoff_key(system: str, model: str) -> str:
    return f"{system}__{model}"


def tune_llm_cutoffs(
    dev_cache: ResultCache,
    systems: Mapping[str, System],
    dev_cases: Sequence[str],
    gold: Mapping[str, frozenset[str]],
) -> dict[str, dict[str, Any]]:
    """Tune each LLM system's confidence cut-off on its cached dev results (micro F1).

    Only results under the system's current configuration count, and only a complete dev
    run (every frozen dev case answered at least once) is tuned; otherwise ``tau`` is None
    and no cut-off is applied. Repeats are pooled.
    """
    wanted = set(dev_cases)
    out: dict[str, dict[str, Any]] = {}
    for name in LLM_SYSTEMS:
        system = systems.get(name)
        if system is None:
            continue
        config_hash = sha256_json(system.config())
        for model in dev_cache.models(name):
            records = [
                r
                for r in dev_cache.load(name, model).values()
                if r["config_hash"] == config_hash and r["ev_id"] in wanted
            ]
            if not records:
                continue
            done = {r["ev_id"] for r in records if r["repeat"] == 0}
            entry: dict[str, Any] = {
                "system": name,
                "model": model,
                "dev_cases": len(wanted),
                "dev_done": len(done),
                "tau": None,
                "dev_micro_f1": None,
                "dev_micro_f1_raw": micro_f1(
                    [(set(r.get("labels") or []), gold[r["ev_id"]]) for r in records]
                ),
                "scores": {},
            }
            if done == wanted:
                tau, scores = tune_confidence(records, gold)
                entry.update(tau=tau, dev_micro_f1=scores[tau], scores=scores)
            out[cutoff_key(name, model)] = entry
    return out


def main(
    argv: list[str] | None = None,
    *,
    provider_factory: ProviderFactory | None = None,
    retriever_factory: RetrieverFactory | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Compare the causal systems.")
    parser.add_argument(
        "--split",
        choices=("held_out", "dev"),
        default="held_out",
        help="dev: run the LLM systems on 2023 history events to tune their cut-offs",
    )
    parser.add_argument("--cases", type=int, default=None, help="sample N cases (default all)")
    parser.add_argument("--refreeze", action="store_true", help="replace the frozen case list")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--limit", type=int, default=None, help="LLM cases to run this time")
    parser.add_argument("--model", default=LLM_MODEL, help="model for the LLM systems")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tune-limit", type=int, default=None, help="2023 events for tuning")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dev-cases-path", type=Path, default=DEFAULT_DEV_CASES)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    dev = args.split == "dev"
    out_json = args.out_json or (DEFAULT_DEV_JSON if dev else DEFAULT_JSON)
    out_md = args.out_md or (DEFAULT_DEV_MD if dev else DEFAULT_MD)

    cache = ResultCache(args.runs_dir)
    dev_cache = ResultCache(args.runs_dir / DEV_RUNS)
    tracker = SpendTracker(max_usd=max_usd())
    make_retriever = retriever_factory or default_retriever_factory()
    workers = Workers(args.database_url, make_retriever)
    stopped: str | None = None
    with connect(args.database_url) as conn:
        scored_rows = scored_held_out_events(conn)
        scored = [ev for ev, _ in scored_rows]
        gold = get_all_gold_labels(conn)
        dates: dict[str, date] = dict(conn.execute("SELECT ev_id, ev_date FROM events").fetchall())
        history_gold = {ev: g for ev, g in gold.items() if dates[ev] < HELD_OUT_FROM}
        taxonomy = load_taxonomy(conn)
        tuning = tuning_events(conn, args.tune_limit, args.seed)
        dev_cases = (
            freeze_cases(
                args.dev_cases_path, tuning, args.cases, args.seed, args.refreeze, DEV_DESCRIPTION
            )
            if dev
            else _frozen(args.dev_cases_path)
        )
        cases = (
            dev_cases
            if dev
            else freeze_cases(args.cases_path, scored, args.cases, args.seed, args.refreeze)
        )
        # Each case's own history-only cutoff: its date for a 2023 dev case, 2024-01-01 else.
        cutoffs = {ev: corpus_cutoff(dates[ev]) for ev in {*scored, *cases}}

        prior = FrequencyPrior(history_gold, dates, tuning, HELD_OUT_FROM, TUNING_FROM)
        systems: dict[str, System] = {"frequency_prior": prior}
        retrieval_config = {"k": KNN_K, "mode": KNN_MODE, **model_names()}
        neighbours = NeighbourCache(args.runs_dir, retrieval_config)

        def get_neighbours(c: Connection, ev_id: str) -> list[str]:
            return neighbours.get(c, ev_id, workers.retriever)

        knn_wanted = "knn_vote" in args.systems or bool(cache.models("knn_vote"))
        tau: float | None = None
        tau_scores: dict[float, float] = {}
        if knn_wanted and not args.report_only:
            _run_parallel(lambda ev: get_neighbours(workers.conn(), ev), tuning, args.workers)
        if knn_wanted and all(ev in neighbours.lists for ev in tuning):
            tuning_lists = {ev: neighbours.lists[ev] for ev in tuning}
            tau, tau_scores = tune_tau(tuning_lists, history_gold, 5)
            systems["knn_vote"] = KnnVote(history_gold, tau, get_neighbours, retrieval_config)

        def agent_factory(c: Connection, verify: bool) -> Agent:
            return Agent(c, workers.retriever(c), provider, verify=verify)

        provider: Provider = (provider_factory or _anthropic)(args.model, tracker)
        systems["closed_book"] = ClosedBook(provider, taxonomy)
        systems["random_context"] = RandomContext(provider, agent_factory, args.seed)
        systems["rag"] = Rag(provider, agent_factory)
        systems["rag_verify"] = RagVerify(provider, agent_factory)

        if not args.report_only:
            tasks: list[Task] = []
            llm_cases = cases[: args.limit] if args.limit is not None else cases
            for name in args.systems:
                system = systems.get(name)
                if system is None or (dev and name in FREE_SYSTEMS):
                    continue  # the free systems are tuned on the dev events: nothing to run
                done = (dev_cache if dev else cache).load(name, system.model)
                config_hash = sha256_json(system.config())
                ev_ids = scored if name in FREE_SYSTEMS else llm_cases
                reps = 1 if name in FREE_SYSTEMS else args.repeat
                tasks += [
                    Task(system, ev, rep)
                    for rep in range(reps)
                    for ev in ev_ids
                    if (ev, rep, config_hash) not in done
                ]
            print(f"{len(tasks)} case runs to do; MAX_USD ${tracker.max_usd:.2f}")
            stopped = run_tasks(tasks, workers, dev_cache if dev else cache, tracker, args.workers)
            print(
                f"LLM spend this run: ${tracker.spent_usd:.4f}"
                + (f"; stopped: {stopped}" if stopped else "")
            )

        llm_cutoffs = tune_llm_cutoffs(dev_cache, systems, dev_cases, gold)
        taus = {k: v["tau"] for k, v in llm_cutoffs.items()}
        tuning_report = {
            "n_events": len(tuning),
            "frequency_prior_k": prior.k,
            "frequency_prior_f1": prior.tuning_f1,
            "knn_tau": tau,
            "knn_tau_f1": tau_scores,
            "dev_cases": {"n": len(dev_cases), "path": str(args.dev_cases_path)},
            "llm_cutoffs": llm_cutoffs,
        }
        if dev:
            llm_systems = {n: s for n, s in systems.items() if n in LLM_SYSTEMS}
            run = partial(
                build_rows,
                conn,
                llm_systems,
                dev_cache,
                cases,
                cases,
                gold,
                history_gold,
                cutoffs,
                max(1, args.repeat),
            )
            raw_rows, _, total_cost = run()
            tuned_rows, _, _ = run(taus)
            rows = [r for r in tuned_rows if r["cutoff"] is not None] + raw_rows
            rows += _dev_reference_rows(
                prior, systems.get("knn_vote"), neighbours, cases, gold, history_gold, cutoffs
            )
            report: dict[str, Any] = {
                "split": "dev",
                "git_sha": git_sha(),
                "cases": {
                    "n": len(cases),
                    "n_scored": len(tuning),
                    "path": str(args.dev_cases_path),
                    "seed": args.seed,
                },
                "taxonomy": {
                    "codes": len(taxonomy),
                    "gold_outside_cases": gold_outside({ev: gold[ev] for ev in cases}, taxonomy),
                },
                "tuning": tuning_report,
                "retrieval": retrieval_config,
                "rows": sorted(rows, key=_row_order),
                "total_cost_usd": total_cost,
                "stopped": stopped,
            }
        else:
            rows, answers, total_cost = build_rows(
                conn,
                systems,
                cache,
                cases,
                scored,
                gold,
                history_gold,
                cutoffs,
                max(1, args.repeat),
                taus,
            )
            verify_row = next(
                (r for r in rows if r["system"] == "rag_verify" and r["model"] == args.model),
                None,
            )
            # Sample only from a complete rag_verify run with its tuned cut-off, so early
            # cases are not over-represented and the audit grades the reported answers.
            if verify_row is not None and verify_row["complete"] and verify_row["cutoff"]:
                write_audit_sample(args.audit, answers[args.model], taxonomy, args.seed)
            verifier_rate = verify_row.get("verifier_support_rate") if verify_row else None
            report = {
                "split": "held_out",
                "git_sha": git_sha(),
                "cases": {
                    "n": len(cases),
                    "n_scored": len(scored),
                    "path": str(args.cases_path),
                    "seed": args.seed,
                },
                "taxonomy": {
                    "codes": len(taxonomy),
                    "gold_outside_cases": gold_outside({ev: gold[ev] for ev in cases}, taxonomy),
                    "gold_outside_all": gold_outside({ev: gold[ev] for ev in scored}, taxonomy),
                },
                "tuning": tuning_report,
                "retrieval": retrieval_config,
                "rows": rows,
                "total_cost_usd": total_cost,
                "citation_audit": audit_summary(args.audit, verifier_rate),
                "stopped": stopped,
            }
    workers.close()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    markdown = render_dev_markdown(report) if dev else render_markdown(report)
    out_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 1 if stopped else 0


def _frozen(path: Path) -> list[str]:
    """The frozen dev cases, or none when the dev split has not been run."""
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8"))["ev_ids"])


def _row_order(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (SYSTEMS.index(row["system"]), row["model"], row["cutoff"] is None)


def _dev_reference_rows(
    prior: FrequencyPrior,
    knn: System | None,
    neighbours: NeighbourCache,
    cases: Sequence[str],
    gold: Mapping[str, frozenset[str]],
    history_gold: Mapping[str, frozenset[str]],
    cutoffs: Mapping[str, date],
) -> list[dict[str, Any]]:
    """The free systems on the dev cases, for reference: they were tuned on these events.

    The frequency prior uses its pre-2023 list; the kNN vote uses the cached neighbour
    lists (each searched before its case's date) and the tuned cut-off. No retrieval runs.
    """
    rows = []
    results = [{"ev_id": ev, "labels": prior.tuning_labels} for ev in cases]
    if isinstance(knn, KnnVote) and all(ev in neighbours.lists for ev in cases):
        knn_results = [
            {
                "ev_id": ev,
                "labels": knn_predict(neighbours.lists[ev], history_gold, knn.tau, 5),
            }
            for ev in cases
        ]
    else:
        knn_results = []
    for name, res in (("frequency_prior", results), ("knn_vote", knn_results)):
        if not res:
            continue
        row = score_cases(res, gold, history_gold=history_gold, passages={}, cutoffs=cutoffs)
        row.update(
            system=name,
            model="none",
            cutoff=None,
            scope="dev (tuned on these)",
            n_expected=len(cases),
            complete=True,
        )
        rows.append(row)
    return rows


def _anthropic(model: str, tracker: SpendTracker) -> Provider:
    from causetrace.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(model, tracker=tracker)


def _run_parallel(fn: Callable[[str], Any], items: Sequence[str], n_workers: int) -> None:
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        for future in [pool.submit(fn, it) for it in items]:
            future.result()
