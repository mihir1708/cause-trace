"""Load-test summary: parsing Locust's stats CSV and rendering the report (synthetic CSV)."""

import json
from pathlib import Path

import pytest

from causetrace.evaluation.load_report import read_locust_stats, write_report

HEADER = (
    "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
    "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
    "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%"
)


def _csv(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_reads_the_aggregated_row_and_writes_the_report(tmp_path: Path) -> None:
    stats = _csv(
        tmp_path / "u10_stats.csv",
        [
            "POST,/v1/search,200,4,90,95,40,300,5000,3.3,0.07,90,100,110,120,150,180,210,250,"
            "290,300,300",
            ",Aggregated,200,4,90,95,40,300,5000,3.3,0.07,90,100,110,120,150,180,210,250,"
            "290,300,300",
        ],
    )
    level = read_locust_stats(stats, 10)
    assert level == {
        "users": 10, "requests": 200, "failures": 4, "rps": 3.3, "p50_ms": 90.0,
        "p95_ms": 180.0, "p99_ms": 250.0, "error_rate": pytest.approx(0.02),
    }  # fmt: skip
    out_json, out_md = tmp_path / "load.json", tmp_path / "load.md"
    write_report([level], host="http://api", duration="60s", out_json=out_json, out_md=out_md)
    assert json.loads(out_json.read_text())["levels"][0]["users"] == 10
    assert "| 10 | 200 | 3.30 | 90 | 180 | 250 | 2.00% |" in out_md.read_text()
