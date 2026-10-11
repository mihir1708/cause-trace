"""Retrieval evaluation runner: known-item questions and held-out incidents, the results
report and the CI baseline gate.

Held-out queries are built from the incident packet only (``causetrace.packets``), and
every search uses ``before_date = corpus_cutoff(event date)`` and excludes the event, so
the corpus is history only (docs/EVAL_PROTOCOL.md, docs/DATA_LEAKAGE.md).
"""

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from causetrace.config import EMBEDDING_MODEL, EMBEDDING_REVISION, RERANK_MODEL, RERANK_REVISION
from causetrace.db import Connection, connect
from causetrace.evaluation import metrics as m
from causetrace.evaluation.questions import kept_questions, read_jsonl
from causetrace.gold import get_all_gold_labels
from causetrace.packets import IncidentPacket, get_incident_packet
from causetrace.retrieval import MODES, Mode, Retriever
from causetrace.split import HELD_OUT_FROM, corpus_cutoff

DEFAULT_QUESTIONS = Path("evals/questions_reviewed.jsonl")
DEFAULT_JSON = Path("artifacts/retrieval_results.json")
DEFAULT_MD = Path("artifacts/retrieval_results.md")
DEFAULT_BASELINE = Path("artifacts/ci_baseline.json")
DEFAULT_TOLERANCE = 0.02
GATE_MODE: Mode = "rrf_rerank"
MANIFEST = Path("data_manifest.json")

_SCORED_HELD_OUT_SQL = """
SELECT e.ev_id, e.ev_date
FROM events e
WHERE e.ev_date >= %s
  AND EXISTS (SELECT 1 FROM findings f WHERE f.ev_id = e.ev_id AND f.cm_inpc IS TRUE)
  AND EXISTS (SELECT 1 FROM narratives n
              WHERE n.ev_id = e.ev_id AND coalesce(btrim(n.narr_accp), '') <> '')
ORDER BY e.ev_date, e.ev_id
"""


def scored_held_out_events(conn: Connection) -> list[tuple[str, date]]:
    """Held-out events with gold labels and a non-empty factual narrative."""
    return [(r[0], r[1]) for r in conn.execute(_SCORED_HELD_OUT_SQL, (HELD_OUT_FROM,))]


def packet_query_text(packet: IncidentPacket) -> str:
    """The query for a target event: its gated factual narratives, else its SAFE metadata."""
    narratives = [a.narr_accp for a in packet.aircraft if a.narr_accp]
    if narratives:
        return "\n\n".join(narratives)
    parts = []
    for a in packet.aircraft:
        parts.append(" ".join(x for x in (a.category, a.make, a.model, a.damage) if x))
    w = packet.weather
    parts.append(" ".join(x for x in (w.basic_conditions, w.light_condition) if x))
    return ". ".join(p for p in parts if p)


def select_events(
    events: list[tuple[str, date]], limit: int | None, seed: int
) -> list[tuple[str, date]]:
    """All events, or ``limit`` of them sampled with ``random.Random(seed)`` (kept in order)."""
    if limit is None or limit >= len(events):
        return events
    chosen = set(random.Random(seed).sample(range(len(events)), limit))
    return [e for i, e in enumerate(events) if i in chosen]


def _summary(values: dict[str, list[float]], latencies: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {name: m.mean(v) for name, v in values.items()}
    out["p50_ms"] = m.percentile(latencies, 50)
    out["p95_ms"] = m.percentile(latencies, 95)
    return out


def evaluate_incidents(
    conn: Connection,
    retriever: Retriever,
    events: Sequence[tuple[str, date]],
    modes: Sequence[Mode],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Per-mode means of Label Recall@10, nDCG@10, MRR, Hit@10 and latency percentiles."""
    labels = get_all_gold_labels(conn)
    dates: dict[str, date] = dict(conn.execute("SELECT ev_id, ev_date FROM events").fetchall())
    pools: dict[date, Counter[frozenset[str]]] = {}
    queries = []
    for ev_id, ev_date in events:
        before = corpus_cutoff(ev_date)
        if before not in pools:
            pools[before] = Counter(ls for ev, ls in labels.items() if dates[ev] < before)
        queries.append((ev_id, before, packet_query_text(get_incident_packet(ev_id, conn))))

    results: dict[str, Any] = {}
    for mode in modes:
        values: dict[str, list[float]] = {
            "label_recall_at_10": [],
            "ndcg_at_10": [],
            "mrr": [],
            "hit_at_10": [],
        }
        latencies: list[float] = []
        for i, (ev_id, before, text) in enumerate(queries, start=1):
            gold = labels[ev_id]
            start = time.perf_counter()
            result = retriever.search_events(
                text, m.K, before_date=before, exclude_event_id=ev_id, mode=mode
            )
            latencies.append((time.perf_counter() - start) * 1000)
            ranked = [e.event_id for e in result.events]
            if ev_id in ranked or any(dates[e] >= before for e in ranked):
                raise AssertionError(f"leakage: {mode} returned the target or a late event")
            rank = m.first_relevant_rank(gold, ranked, labels)
            values["label_recall_at_10"].append(m.label_recall_at_k(gold, ranked, labels))
            values["ndcg_at_10"].append(m.ndcg_at_k(gold, ranked, labels, pools[before]))
            values["mrr"].append(m.reciprocal_rank(rank))
            values["hit_at_10"].append(1.0 if rank else 0.0)
            if progress and i % 50 == 0:
                progress(f"{mode}: {i}/{len(queries)}")
        results[mode] = _summary(values, latencies)
    return {"n_queries": len(queries), "modes": results}


def evaluate_questions(
    retriever: Retriever,
    questions: Sequence[dict[str, Any]],
    modes: Sequence[Mode],
    known_events: set[str],
) -> dict[str, Any]:
    """Known-item Recall@10 (source event in the top 10 distinct events), MRR and p95.

    The corpus is history only (before HELD_OUT_FROM); the source event is not excluded.
    """
    usable = [q for q in questions if q["event_id"] in known_events]
    results: dict[str, Any] = {}
    for mode in modes:
        hits: list[float] = []
        rrs: list[float] = []
        latencies: list[float] = []
        for q in usable:
            start = time.perf_counter()
            result = retriever.search_events(
                q["question"], m.K, before_date=HELD_OUT_FROM, mode=mode
            )
            latencies.append((time.perf_counter() - start) * 1000)
            rank = m.known_item_rank(q["event_id"], [e.event_id for e in result.events])
            hits.append(1.0 if rank else 0.0)
            rrs.append(m.reciprocal_rank(rank))
        results[mode] = {
            "recall_at_10": m.mean(hits),
            "mrr": m.mean(rrs),
            "p95_ms": m.percentile(latencies, 95),
        }
    return {
        "n_queries": len(usable),
        "n_skipped_unknown_event": len(questions) - len(usable),
        "modes": results,
    }


# --- Provenance and report ------------------------------------------------------


def git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def snapshot_sha256(manifest: Path = MANIFEST) -> str | None:
    """SHA-256 of the NTSB download (avall.zip) recorded in data_manifest.json."""
    if not manifest.is_file():
        return None
    sha = json.loads(manifest.read_text(encoding="utf-8")).get("files", {}).get("avall.zip", {})
    value = sha.get("sha256")
    return str(value) if value else None


def model_names() -> dict[str, str]:
    return {
        "embedding": f"{EMBEDDING_MODEL}@{EMBEDDING_REVISION}",
        "rerank": f"{RERANK_MODEL}@{RERANK_REVISION}",
    }


def dataset_counts(conn: Connection) -> dict[str, int]:
    row = conn.execute(
        "SELECT count(*), count(*) FILTER (WHERE ev_date < %s), "
        "(SELECT count(*) FROM passages) FROM events",
        (HELD_OUT_FROM,),
    ).fetchone()
    assert row is not None
    return {
        "events": row[0],
        "history_events": row[1],
        "held_out_events": row[0] - row[1],
        "passages": row[2],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    q, inc = report["questions"], report["incidents"]
    lines = [
        "# Retrieval results",
        "",
        "Generated by `scripts/eval_retrieval.py`; do not edit by hand.",
        "",
        f"- Git SHA: `{report['git_sha']}`",
        f"- Embedding model: `{report['models']['embedding']}`",
        f"- Rerank model: `{report['models']['rerank']}`",
        f"- Snapshot SHA-256 (avall.zip, data_manifest.json): `{report['snapshot_sha256']}`",
        f"- Database: {report['dataset']['events']} events "
        f"({report['dataset']['history_events']} history, "
        f"{report['dataset']['held_out_events']} held-out), "
        f"{report['dataset']['passages']} passages",
        f"- Settings: top {m.K} distinct events; relevance J >= {m.RELEVANCE_THRESHOLD}; "
        f"nearest-rank percentiles; limit {report['settings']['limit']}, "
        f"seed {report['settings']['seed']}",
        "",
        f"## Reviewed questions (known item), N = {q['n_queries']}",
        "",
    ]
    if q["n_queries"]:
        lines += ["| Mode | Recall@10 | MRR | p95 ms |", "|---|---|---|---|"]
        for mode, r in q["modes"].items():
            lines.append(
                f"| {mode} | {_fmt(r['recall_at_10'])} | {_fmt(r['mrr'])} "
                f"| {_fmt(r['p95_ms'], 0)} |"
            )
    else:
        lines.append(f"Not run: no reviewed questions in `{q['source']}`.")
    lines += [
        "",
        f"## Held-out incidents, N = {inc['n_queries']}",
        "",
        "| Mode | Label Recall@10 | nDCG@10 | MRR | Hit@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, r in inc["modes"].items():
        lines.append(
            f"| {mode} | {_fmt(r['label_recall_at_10'])} | {_fmt(r['ndcg_at_10'])} "
            f"| {_fmt(r['mrr'])} | {_fmt(r['hit_at_10'])} | {_fmt(r['p50_ms'], 0)} "
            f"| {_fmt(r['p95_ms'], 0)} |"
        )
    return "\n".join(lines) + "\n"


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", "utf-8")


# --- CI gate --------------------------------------------------------------------


def query_set_sha256(events: Sequence[tuple[str, date]]) -> str:
    return hashlib.sha256("\n".join(sorted(ev for ev, _ in events)).encode()).hexdigest()


def baseline_record(
    incidents: dict[str, Any], events: Sequence[tuple[str, date]]
) -> dict[str, Any]:
    return {
        "mode": GATE_MODE,
        "label_recall_at_10": incidents["modes"][GATE_MODE]["label_recall_at_10"],
        "n_queries": incidents["n_queries"],
        "query_set_sha256": query_set_sha256(events),
        "models": model_names(),
        "git_sha": git_sha(),
        "generated_by": "make update-baseline",
    }


def check_gate(
    current: dict[str, Any], baseline: dict[str, Any], tolerance: float = DEFAULT_TOLERANCE
) -> tuple[bool, str]:
    """Fail when Label Recall@10 drops by more than ``tolerance`` or the query set differs."""
    if current["query_set_sha256"] != baseline["query_set_sha256"]:
        return False, (
            "query set differs from the baseline (fixture or eligibility changed); "
            "run `make update-baseline` deliberately and commit the result"
        )
    drop = baseline["label_recall_at_10"] - current["label_recall_at_10"]
    msg = (
        f"{GATE_MODE} Label Recall@10: baseline {baseline['label_recall_at_10']:.4f}, "
        f"current {current['label_recall_at_10']:.4f}, drop {drop:+.4f} "
        f"(tolerance {tolerance:.2f}, N={current['n_queries']})"
    )
    return drop <= tolerance + 1e-12, msg


# --- CLI ------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    retriever_factory: Callable[[Connection], Retriever] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (docs/EVAL_PROTOCOL.md).")
    parser.add_argument("--limit", type=int, default=None, help="evaluate N held-out events")
    parser.add_argument("--seed", type=int, default=42, help="seed for --limit sampling")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--gate", type=Path, default=None, help="compare rrf_rerank with baseline")
    parser.add_argument("--write-baseline", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)

    with connect(args.database_url) as conn:
        retriever = retriever_factory(conn) if retriever_factory else Retriever(conn)
        events = select_events(scored_held_out_events(conn), args.limit, args.seed)

        if args.gate or args.write_baseline:
            incidents = evaluate_incidents(conn, retriever, events, [GATE_MODE], print)
            current = baseline_record(incidents, events)
            print(json.dumps(current, indent=2, sort_keys=True))
            if args.write_baseline:
                write_json(current, args.write_baseline)
                print(f"wrote {args.write_baseline}")
            if args.gate:
                if not args.gate.is_file():
                    print(f"FAIL: no baseline at {args.gate}; run `make update-baseline`")
                    return 1
                ok, msg = check_gate(
                    current, json.loads(args.gate.read_text("utf-8")), args.tolerance
                )
                print(("PASS: " if ok else "FAIL: ") + msg)
                return 0 if ok else 1
            return 0

        modes: list[Mode] = list(args.modes)
        known = {
            r[0]
            for r in conn.execute("SELECT ev_id FROM events WHERE ev_date < %s", (HELD_OUT_FROM,))
        }
        questions = evaluate_questions(
            retriever, kept_questions(read_jsonl(args.questions)), modes, known
        )
        questions["source"] = str(args.questions)
        incidents = evaluate_incidents(conn, retriever, events, modes, print)
        report = {
            "git_sha": git_sha(),
            "models": model_names(),
            "snapshot_sha256": snapshot_sha256(),
            "dataset": dataset_counts(conn),
            "settings": {"limit": args.limit, "seed": args.seed, "k": m.K},
            "questions": questions,
            "incidents": incidents,
        }
    write_json(report, args.out_json)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0
