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
        "levels": [{"users": 1, "rps": 2.5, "p50_ms": 30.0, "p95_ms": 50.0, "p99_ms": 70.0,
                    "error_rate": 0.0}],
    }  # fmt: skip
    (tmp_path / "load_results.json").write_text(json.dumps(report))
    text = "a\n<!-- results:load -->\nold\n<!-- /results:load -->\nb\n"
    out = render(text, {"load": blocks(tmp_path)["load"]})
    assert "| 1 | 2.50 | 30 | 50 | 70 | 0.00% |" in out and "old" not in out
    assert render(out, {"load": blocks(tmp_path)["load"]}) == out  # idempotent
    with pytest.raises(ValueError, match="results:causal"):
        render(text, {"causal": "x"})
