"""README results are generated from artifacts/ and kept up to date (no hand-typed numbers)."""

import json
from pathlib import Path

import pytest

from causetrace.evaluation.readme import ARTIFACTS, README, blocks, load_block, render

ROOT = Path(__file__).resolve().parents[1]


def test_readme_results_match_the_artifacts() -> None:
    current = (ROOT / README).read_text(encoding="utf-8")
    assert render(current, blocks(ROOT / ARTIFACTS)) == current, (
        "README.md results are stale: run `make readme`"
    )


def test_load_block_and_markers(tmp_path: Path) -> None:
    assert "Not run yet" in load_block(None)
    report = {
        "git_sha": "abc123", "endpoint": "POST /v1/search", "duration": "60s",
        "levels": [{"users": 1, "requests": 150, "failures": 0, "rps": 2.5, "p50_ms": 30.0,
                    "p95_ms": 50.0, "p99_ms": 70.0, "error_rate": 0.0}],
    }  # fmt: skip
    (tmp_path / "load_results.json").write_text(json.dumps(report))
    text = "a\n<!-- results:load -->\nold\n<!-- /results:load -->\nb\n"
    out = render(text, {"load": blocks(tmp_path)["load"]})
    assert "| 1 | 2.5 | 0.0 s | 0.1 s | 0.0% |" in out and "old" not in out
    assert render(out, {"load": blocks(tmp_path)["load"]}) == out  # idempotent
    with pytest.raises(ValueError, match="results:causal"):
        render(text, {"causal": "x"})


def test_findings_wording_follows_the_data() -> None:
    from causetrace.evaluation.readme import findings_block

    def row(system: str, model: str, f1: float) -> dict[str, object]:
        return {"system": system, "model": model, "scope": "cases", "micro_f1": f1}

    causal = {
        "rows": [row("knn_vote", "none", 0.2), row("rag", "claude-haiku-5-5", 0.4)],
        "cases": {"n": 10},
        "total_cost_usd": 1.0,
    }
    text = findings_block(None, causal, None)
    assert "Claude leads" in text and "`rag`" in text and "hard to beat" not in text
    causal["rows"] = [row("knn_vote", "none", 0.5), row("rag", "claude-haiku-5-5", 0.4)]
    assert "hard to beat" in findings_block(None, causal, None)


def test_example_shows_official_findings_only_for_the_picked_case(tmp_path: Path) -> None:
    from causetrace.evaluation.readme import example_block

    art = tmp_path / "artifacts"
    (art / "runs").mkdir(parents=True)
    (art / "cases.json").write_text(json.dumps({"ev_ids": ["e1", "e2"]}))

    def finding(label: str, conf: float) -> dict[str, object]:
        return {"label": label, "short_name": f"A-{label}", "confidence": conf,
                "citations": [f"p{label}:factual:0"], "rationale": "r"}  # fmt: skip

    records = [
        {"ev_id": "e2", "repeat": 0, "cites": True, "findings": [finding("01", 0.9),
                                                                   finding("02", 0.8)]},
        {"ev_id": "e1", "repeat": 0, "cites": True, "findings": [finding("03", 0.9),
                                                                   finding("04", 0.1)]},
    ]  # fmt: skip
    (art / "runs" / "rag_verify__claude-haiku-5-5.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )
    causal = {
        "tuning": {"llm_cutoffs": {"rag_verify__claude-haiku-5-5": {"tau": 0.5}}},
        "cases": {"path": "artifacts/cases.json"},
    }
    # e1 keeps one finding after the cut-off, so the rule picks e2.
    text = example_block(causal, tmp_path)
    assert "`e2`" in text and "NTSB concluded" not in text
    (art / "readme_example.json").write_text(
        json.dumps({"ev_id": "e1", "official": [{"label": "03", "description": "X"}]})
    )
    assert "NTSB concluded" not in example_block(causal, tmp_path)  # another case: ignored
    (art / "readme_example.json").write_text(
        json.dumps({"ev_id": "e2", "official": [{"label": "02", "description": "Not determined"}]})
    )
    text = example_block(causal, tmp_path)
    assert "Also an NTSB finding?" in text and "| Yes |" in text and "| No |" in text
    assert "NTSB concluded for this case: Not determined. 1 of 2 suggested causes" in text
