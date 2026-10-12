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
