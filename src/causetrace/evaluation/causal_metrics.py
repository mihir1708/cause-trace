"""Causal-system metrics (docs/EVAL_PROTOCOL.md, "Causal metrics").

A case result is a dict with ``ev_id``, ``labels`` (predicted codes), ``abstained``,
optional ``citations`` (``{"label", "passage_id"}``), ``context_passages`` (passage ids the
system was given), optional ``verifier`` (``{"pairs", "supported"}``) and usage fields
(``tokens``, ``calls``, ``seconds``, ``cost_usd``).
"""

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class PassageInfo:
    event_id: str
    event_date: date


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_cases(
    results: Sequence[Mapping[str, Any]],
    gold: Mapping[str, frozenset[str]],
    *,
    history_gold: Mapping[str, frozenset[str]],
    passages: Mapping[str, PassageInfo],
    cutoffs: Mapping[str, date],
) -> dict[str, Any]:
    """Metrics for one system run over ``results`` (one per case).

    - Micro P/R/F1 with abstentions as empty predictions; macro = per-case means, where an
      empty prediction has precision 0. An explicit abstention and an empty prediction are
      both abstentions.
    - ``micro_f1_answered``: micro F1 over cases that were not abstentions.
    - Citation validity: the cited passage was in the context the system was given and is
      from a history event dated before the case's cutoff.
    - Citation support: the cited passage's event has the cited label among its gold labels
      (invalid citations count as unsupported).
    """
    tp = fp = fn = 0
    tp_a = fp_a = fn_a = 0
    per_p: list[float] = []
    per_r: list[float] = []
    per_f: list[float] = []
    citations = valid = supported = 0
    pairs = verified = 0
    has_verifier = False
    abstained = 0
    for res in results:
        g = gold[res["ev_id"]]
        p = set(res.get("labels") or [])
        if res.get("abstained") or not p:  # an empty answer counts as an abstention
            abstained += 1
            p = set()
        ctp, cfp, cfn = len(p & g), len(p - g), len(g - p)
        tp, fp, fn = tp + ctp, fp + cfp, fn + cfn
        if p:
            tp_a, fp_a, fn_a = tp_a + ctp, fp_a + cfp, fn_a + cfn
        cp, cr, cf = _prf(ctp, cfp, cfn)
        per_p.append(cp)
        per_r.append(cr)
        per_f.append(cf)

        context = set(res.get("context_passages") or [])
        cutoff = cutoffs[res["ev_id"]]
        for c in res.get("citations") or []:
            citations += 1
            info = passages.get(c["passage_id"])
            ok = info is not None and c["passage_id"] in context and info.event_date < cutoff
            valid += ok
            if ok and info is not None and c["label"] in history_gold.get(info.event_id, ()):
                supported += 1
        v = res.get("verifier")
        if v is not None:
            has_verifier = True
            pairs += int(v.get("pairs", 0))
            verified += int(v.get("supported", 0))

    n = len(results)
    micro_p, micro_r, micro_f = _prf(tp, fp, fn)
    _, _, f_answered = _prf(tp_a, fp_a, fn_a)
    cites = any(res.get("cites") for res in results)
    return {
        "n_cases": n,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f,
        "micro_f1_answered": f_answered if n - abstained else None,
        "macro_precision": _mean(per_p),
        "macro_recall": _mean(per_r),
        "macro_f1": _mean(per_f),
        "recovered": tp,
        "gold_findings": tp + fn,
        "predicted": tp + fp,
        "abstention_rate": abstained / n if n else None,
        "citations": citations if cites else None,
        "citation_validity": valid / citations if cites and citations else None,
        "citation_support": supported / citations if cites and citations else None,
        "verifier_pairs": pairs if has_verifier else None,
        "verifier_support_rate": verified / pairs if has_verifier and pairs else None,
        "mean_tokens": _mean([float(r.get("tokens", 0)) for r in results]),
        "mean_calls": _mean([float(r.get("calls", 0)) for r in results]),
        "mean_seconds": _mean([float(r.get("seconds", 0.0)) for r in results]),
        "mean_cost_usd": _mean([float(r.get("cost_usd", 0.0)) for r in results]),
        "total_cost_usd": sum(float(r.get("cost_usd", 0.0)) for r in results),
    }


def aggregate_repeats(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean of each numeric metric over repeats, plus its min and max (the spread)."""
    if len(runs) == 1:
        return {**runs[0], "repeats": 1}
    out: dict[str, Any] = {"repeats": len(runs)}
    for key in runs[0]:
        values = [r[key] for r in runs if isinstance(r.get(key), int | float)]
        if len(values) == len(runs):
            out[key] = statistics.fmean(values)
            out[f"{key}_min"] = min(values)
            out[f"{key}_max"] = max(values)
        else:
            out[key] = runs[0].get(key)
    return out


def gold_outside(gold: Mapping[str, frozenset[str]], taxonomy: Iterable[str]) -> dict[str, int]:
    """Gold labels the systems cannot predict because they are not in the taxonomy."""
    codes = set(taxonomy)
    outside = {ev: g - codes for ev, g in gold.items()}
    return {
        "labels": sum(len(o) for o in outside.values()),
        "cases": sum(bool(o) for o in outside.values()),
        "distinct_codes": len(set().union(*outside.values())) if outside else 0,
    }
