"""Results sections of README.md, generated from artifacts/ (``scripts/render_readme.py``).

Each block sits between ``<!-- results:<name> -->`` and ``<!-- /results:<name> -->``
markers; everything between them is replaced. No number in those blocks is typed by hand.
The wording is plain on purpose: the README is read by people outside the field.
"""

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from causetrace.evaluation.causal_systems import apply_cutoff

ARTIFACTS = Path("artifacts")
README = Path("README.md")
BLOCKS = ("findings", "example", "retrieval", "causal", "load")

SEARCH_MODES = {
    "dense": "Meaning-based search (embeddings)",
    "lexical": "Keyword search (PostgreSQL full-text)",
    "rrf": "Both combined (reciprocal rank fusion)",
    "rrf_rerank": "Both combined, then reranked by a cross-encoder",
}
SYSTEMS = {
    "frequency_prior": "Always guesses the most common causes (no AI)",
    "knn_vote": "Vote of the 10 most similar past accidents (no AI)",
    "closed_book": "Claude with the accident facts only",
    "random_context": "Claude with 10 random past reports (control)",
    "rag": "Claude with the most similar past reports",
    "rag_verify": "Same, plus a step that checks every citation",
}


def _f(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{100 * value:.{digits}f}%"


def _rows(c: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """One row per system: the all-events row for the free systems, else the cases row."""
    out: dict[str, dict[str, Any]] = {}
    for row in c["rows"]:
        if row.get("scope") not in ("all", "cases"):
            continue
        if row["system"] not in out or row["scope"] == "all":
            out[row["system"]] = row
    return out


def findings_block(
    retrieval: Mapping[str, Any] | None,
    causal: Mapping[str, Any] | None,
    load: Mapping[str, Any] | None,
) -> str:
    bullets: list[str] = []
    if retrieval:
        modes = retrieval["incidents"]["modes"]
        best = max(modes, key=lambda m: modes[m]["label_recall_at_10"])
        bullets.append(
            f"**Search finds relevant precedent.** For a new accident, the 10 most similar "
            f"earlier investigations found by the search already contain "
            f"{_pct(modes[best]['label_recall_at_10'])} of its official causes on average "
            f"({retrieval['incidents']['n_queries']} accidents from 2024 onward)."
        )
    if causal:
        rows = _rows(causal)
        best = max(rows.values(), key=lambda r: r["micro_f1"])
        llm = [r for r in rows.values() if r["model"] != "none"]
        if llm:
            best_llm = max(llm, key=lambda r: r["micro_f1"])
            what = SYSTEMS.get(best["system"], "").replace(" (no AI)", "")
            what = what[:1].lower() + what[1:]
            if best["model"] == "none":
                bullets.append(
                    f"**A simple baseline is hard to beat.** The best accuracy (F1 "
                    f"{_f(best['micro_f1'], 2)}) comes from `{best['system']}`, a {what} "
                    "with no AI, "
                    f"ahead of the best Claude setup (F1 {_f(best_llm['micro_f1'], 2)}). "
                    "The project reports this rather than hiding it."
                )
            else:
                bullets.append(
                    f"**Claude leads on accuracy.** The best setup is `{best['system']}` "
                    f"({what}), with F1 {_f(best['micro_f1'], 2)}."
                )
        rag, rand = rows.get("rag"), rows.get("random_context")
        if rag and rand and rag.get("citation_support") and rand.get("citation_support"):
            bullets.append(
                f"**Retrieval makes the answers checkable.** With the most similar past reports, "
                f"{_pct(rag['citation_support'])} of Claude's citations point to a report with the "
                f"same cause, against {_pct(rand['citation_support'])} when given random reports."
            )
        audit = causal.get("citation_audit")
        if audit and audit.get("graded"):
            bullets.append(
                f"**Citations hold up to a manual check.** Of {audit['graded']} sampled citations, "
                f"{_pct(audit['precision_strict'])} fully support the predicted cause and "
                f"{_pct(audit['precision_lenient'])} support it at least partly."
            )
        n_llm = sum(r["model"] != "none" for r in rows.values())
        bullets.append(
            f"**Cheap to run.** Running all {n_llm} Claude setups on the "
            f"{causal['cases']['n']} test accidents cost ${causal['total_cost_usd']:.2f} in "
            "API calls."
        )
    if load:
        levels = load["levels"]
        peak = max(levels, key=lambda lv: lv["rps"])
        top = max(levels, key=lambda lv: lv["users"])
        errors = sum(lv["failures"] for lv in levels)
        bullets.append(
            f"**Stable under load.** The search API answered every request ({errors} errors) "
            f"with up to {top['users']} simultaneous users, peaking at "
            f"{peak['rps']:.1f} requests per second on one laptop process."
        )
    return "\n".join(f"- {b}" for b in bullets) or "Results not generated yet."


EXAMPLE_RUNS = Path("runs") / "rag_verify__claude-haiku-5-5.jsonl"
# The NTSB's official findings for the example case, written by
# scripts/export_readme_example.py (needs the database; GOLD_ONLY, for display only).
EXAMPLE_OFFICIAL = Path("readme_example.json")


def example_candidates(causal: Mapping[str, Any] | None, root: Path) -> list[dict[str, Any]]:
    """Held-out rag_verify answers, in ev_id order, with two or more findings after the tuned
    cut-off: the candidates for the README example, in the order the fixed rule tries them."""
    path = root / ARTIFACTS / EXAMPLE_RUNS
    if causal is None or not path.is_file():
        return []
    cut = causal["tuning"]["llm_cutoffs"].get("rag_verify__claude-haiku-5-5", {}).get("tau")
    cases = set(json.loads((root / causal["cases"]["path"]).read_text())["ev_ids"])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    answers = sorted(
        (apply_cutoff(r, cut) for r in records if r["ev_id"] in cases and r["repeat"] == 0),
        key=lambda r: r["ev_id"],
    )
    return [r for r in answers if len(r["findings"]) >= 2]


def pick_example(
    causal: Mapping[str, Any] | None, root: Path, skip: frozenset[str] = frozenset()
) -> dict[str, Any] | None:
    """The first candidate not in ``skip``. A fixed rule: the pick never depends on whether the
    answer is right. ``skip`` holds cases whose only NTSB finding is "not determined" (written
    by scripts/export_readme_example.py), which leave nothing to compare the answer with."""
    return next((r for r in example_candidates(causal, root) if r["ev_id"] not in skip), None)


def example_block(causal: Mapping[str, Any] | None, root: Path) -> str:
    """One real answer from the evaluation run, picked by a fixed rule (not for correctness),
    shown next to the NTSB's official findings when they have been exported."""
    if causal is None or not (root / ARTIFACTS / EXAMPLE_RUNS).is_file():
        return "Not generated yet."
    official = _load(root / ARTIFACTS / EXAMPLE_OFFICIAL)
    skip = frozenset(official.get("skipped_undetermined", [])) if official else frozenset()
    pick = pick_example(causal, root, skip)
    if pick is None:
        return "No example available."
    if official is not None and official.get("ev_id") != pick["ev_id"]:
        official = None  # exported for another case: stale, so not shown
    gold = {o["label"] for o in official["official"]} if official else None
    out = [
        f"NTSB case `{pick['ev_id']}` (the first 2024+ case, in ID order, with two or more "
        "cited findings and a cause the NTSB determined; not selected for being correct). "
        "Each suggested cause comes with the "
        "earlier investigation it relies on:",
        "",
    ]
    header = "| Suggested cause | Confidence | Why (Claude's rationale) | Cited earlier report |"
    rule = "|---|---|---|---|"
    if gold is not None:
        header += " Also an NTSB finding? |"
        rule += "---|"
    out += [header, rule]
    for f in pick["findings"]:
        cited = ", ".join(f"`{c.split(':')[0]}`" for c in f["citations"])
        rationale = str(f.get("rationale") or "").replace("|", "/")
        row = (
            f"| {str(f.get('short_name') or f['label']).split('-')[-1]} "
            f"| {_f(f.get('confidence'), 2)} | {rationale} | {cited} |"
        )
        if gold is not None:
            row += f" {'Yes' if f['label'] in gold else 'No'} |"
        out.append(row)
    if official is not None:
        names = "; ".join(
            str(o.get("description") or o["label"]).replace("|", "/") for o in official["official"]
        )
        hits = sum(f["label"] in gold for f in pick["findings"]) if gold else 0
        out += [
            "",
            f"What the NTSB concluded for this case: {names or 'no cause findings'}. "
            f"{hits} of {len(pick['findings'])} suggested causes match it.",
        ]
    return "\n".join(out)


def retrieval_block(r: Mapping[str, Any]) -> str:
    inc, q = r["incidents"]["modes"], r["questions"]["modes"]
    out = [
        "| Search method | Causes found in top 10 ¹ | Right report in top 10 ² |",
        "|---|---|---|",
    ]
    for mode, label in SEARCH_MODES.items():
        if mode in inc:
            out.append(
                f"| {label} | {_pct(inc[mode]['label_recall_at_10'], 1)} "
                f"| {_pct(q.get(mode, {}).get('recall_at_10'), 1)} |"
            )
    out += [
        "",
        f"¹ Share of a new accident's official causes that appear in the 10 earlier "
        f"investigations returned ({r['incidents']['n_queries']} accidents from 2024 onward). "
        f"² Share of {r['questions']['n_queries']} reviewed test questions whose source report "
        "is in the top 10. Full metrics: "
        "[`artifacts/retrieval_results.md`](artifacts/retrieval_results.md).",
    ]
    return "\n".join(out)


def causal_block(c: Mapping[str, Any]) -> str:
    out = [
        "| System | What it is | Precision | Recall | F1 | Supported citations | Cost per case |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in _rows(c).items():
        out.append(
            f"| `{name}` | {SYSTEMS.get(name, '')} | {_f(row['micro_precision'], 2)} "
            f"| {_f(row['micro_recall'], 2)} | {_f(row['micro_f1'], 2)} "
            f"| {_pct(row.get('citation_support'))} | ${_f(row['mean_cost_usd'], 4)} |"
        )
    out += [
        "",
        f"{c['cases']['n']} accidents from 2024 onward, never seen during development. "
        "*Precision*: share of suggested causes that the NTSB also found. *Recall*: share of "
        "the NTSB's causes that were suggested. *F1*: the balance of the two. *Supported "
        "citations*: share of citations whose earlier report has the same cause. Claude runs "
        "are `claude-haiku-5-5`, with confidence thresholds tuned on 2023 accidents only. Full "
        "metrics: [`artifacts/causal_results.md`](artifacts/causal_results.md).",
    ]
    audit = c.get("citation_audit")
    if audit and audit.get("graded"):
        out += [
            "",
            f"Manual citation check: {audit['graded']} of {audit['sampled']} sampled citations "
            "graded (proposed by Claude, reviewed and approved by Mihir Mukhi): "
            f"{_pct(audit['precision_strict'], 1)} fully support the cause, "
            f"{_pct(audit['precision_lenient'], 1)} at least partly. The automatic verifier "
            f"accepted {_pct(audit['verifier_support_rate'], 1)} of the citations it checked.",
        ]
    return "\n".join(out)


def load_block(lr: Mapping[str, Any] | None) -> str:
    if lr is None:
        return "Not run yet: `make loadtest` writes `artifacts/load_results.json`."
    out = [
        "| Simultaneous users | Requests per second | Median response | 95th percentile | Errors |",
        "|---|---|---|---|---|",
    ]
    for lv in lr["levels"]:
        out.append(
            f"| {lv['users']} | {lv['rps']:.1f} | {lv['p50_ms'] / 1000:.1f} s "
            f"| {lv['p95_ms'] / 1000:.1f} s | {_pct(lv['error_rate'], 1)} |"
        )
    out += [
        "",
        f"{lr['endpoint']}, {lr['duration']} per level. Full report: "
        "[`artifacts/load_results.md`](artifacts/load_results.md).",
    ]
    return "\n".join(out)


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def blocks(artifacts: Path = ARTIFACTS) -> dict[str, str]:
    retrieval = _load(artifacts / "retrieval_results.json")
    causal = _load(artifacts / "causal_results.json")
    load = _load(artifacts / "load_results.json")
    return {
        "findings": findings_block(retrieval, causal, load),
        "example": example_block(causal, artifacts.parent),
        "retrieval": retrieval_block(retrieval) if retrieval else "Not run yet.",
        "causal": causal_block(causal) if causal else "Not run yet.",
        "load": load_block(load),
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
