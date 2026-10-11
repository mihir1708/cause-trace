"""Results tables in README.md, generated from artifacts/ (``scripts/render_readme.py``).

Each table sits between ``<!-- results:<name> -->`` and ``<!-- /results:<name> -->``
markers; everything between them is replaced. No number in those blocks is typed by hand.
"""

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ARTIFACTS = Path("artifacts")
README = Path("README.md")
BLOCKS = ("retrieval", "causal", "load")


def _f(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def retrieval_block(r: Mapping[str, Any]) -> str:
    ds, inc, q = r["dataset"], r["incidents"], r["questions"]
    out = [
        f"From `artifacts/retrieval_results.json` (git `{r['git_sha'][:12]}`): "
        f"{ds['passages']} passages from {ds['history_events']} history events; top 10 "
        "distinct events per query.",
        "",
        f"Held-out incidents (N = {inc['n_queries']}, query = the incident packet):",
        "",
        "| Mode | Label Recall@10 | nDCG@10 | MRR | Hit@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, m in inc["modes"].items():
        out.append(
            f"| {mode} | {_f(m['label_recall_at_10'])} | {_f(m['ndcg_at_10'])} "
            f"| {_f(m['mrr'])} | {_f(m['hit_at_10'])} | {_f(m['p50_ms'], 0)} "
            f"| {_f(m['p95_ms'], 0)} |"
        )
    out += [
        "",
        f"Reviewed known-item questions (N = {q['n_queries']}):",
        "",
        "| Mode | Recall@10 | MRR | p95 ms |",
        "|---|---|---|---|",
    ]
    for mode, m in q["modes"].items():
        out.append(f"| {mode} | {_f(m['recall_at_10'])} | {_f(m['mrr'])} | {_f(m['p95_ms'], 0)} |")
    return "\n".join(out)


def causal_block(c: Mapping[str, Any]) -> str:
    cases = c["cases"]
    out = [
        f"From `artifacts/causal_results.json` (git `{c['git_sha'][:12]}`): "
        f"{cases['n']} of {cases['n_scored']} scored held-out events; LLM systems use "
        "claude-haiku-5-5 with confidence cut-offs tuned on 2023 dev events; LLM cost "
        f"${c['total_cost_usd']:.2f}.",
        "",
        "| System | Cut-off | Micro P | Micro R | Micro F1 | F1 answered | Abstain "
        "| Cit. support | $ / case |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in c["rows"]:
        if row.get("scope") not in ("all", "cases"):
            continue
        if (
            row["system"] in ("frequency_prior", "knn_vote")
            and row["scope"] != "all"
            and any(r["system"] == row["system"] and r["scope"] == "all" for r in c["rows"])
        ):
            continue  # the free systems: show the all-events row
        out.append(
            f"| {row['system']} | {_f(row.get('cutoff'), 2)} | {_f(row['micro_precision'])} "
            f"| {_f(row['micro_recall'])} | {_f(row['micro_f1'])} "
            f"| {_f(row['micro_f1_answered'])} | {_f(row['abstention_rate'])} "
            f"| {_f(row.get('citation_support'))} | {_f(row['mean_cost_usd'], 4)} |"
        )
    audit = c.get("citation_audit")
    if audit and audit.get("graded"):
        out += [
            "",
            f"Citation audit (rag_verify): {audit['graded']} of {audit['sampled']} sampled "
            "citations graded (proposed by Claude, reviewed and approved by Mihir Mukhi): "
            f"{_pct(audit['precision_strict'])} support the label (grade 2), "
            f"{_pct(audit['precision_lenient'])} support it at least partly (grade 1 or 2). "
            f"Verifier support rate: {_pct(audit['verifier_support_rate'])}.",
        ]
    return "\n".join(out)


def load_block(lr: Mapping[str, Any] | None) -> str:
    if lr is None:
        return "Not run yet: `make loadtest` writes `artifacts/load_results.json`."
    out = [
        f"From `artifacts/load_results.json` (git `{lr['git_sha'][:12]}`): "
        f"{lr['endpoint']}, {lr['duration']} per level.",
        "",
        "| Users | RPS | p50 ms | p95 ms | p99 ms | Error rate |",
        "|---|---|---|---|---|---|",
    ]
    for lv in lr["levels"]:
        out.append(
            f"| {lv['users']} | {lv['rps']:.2f} | {lv['p50_ms']:.0f} | {lv['p95_ms']:.0f} "
            f"| {lv['p99_ms']:.0f} | {100 * lv['error_rate']:.2f}% |"
        )
    return "\n".join(out)


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def blocks(artifacts: Path = ARTIFACTS) -> dict[str, str]:
    retrieval = _load(artifacts / "retrieval_results.json")
    causal = _load(artifacts / "causal_results.json")
    return {
        "retrieval": retrieval_block(retrieval) if retrieval else "Not run yet.",
        "causal": causal_block(causal) if causal else "Not run yet.",
        "load": load_block(_load(artifacts / "load_results.json")),
    }


def _filler(content: str) -> Callable[[re.Match[str]], str]:
    def fill(m: re.Match[str]) -> str:
        return m.group(1) + content + "\n" + m.group(2)

    return fill


def render(text: str, generated: Mapping[str, str]) -> str:
    """``text`` with every marker block replaced by its generated content."""
    for name, content in generated.items():
        pattern = re.compile(
            rf"(<!-- results:{name} -->\n)(?:.*?\n)?(<!-- /results:{name} -->)", re.DOTALL
        )
        if not pattern.search(text):
            raise ValueError(f"README has no results:{name} markers")
        text = pattern.sub(_filler(content), text)
    return text
