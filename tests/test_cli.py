"""The causetrace CLI: eval commands pass their arguments through; ask runs the agent."""

import json
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from conftest import FIXTURES, reset
from fakes import FakeEmbedder, FakeReranker
from test_api import Responder
from typer.testing import CliRunner

import causetrace.agent.__main__ as agent_main
import causetrace.evaluation.causal_eval as causal_eval
import causetrace.evaluation.retrieval_eval as retrieval_eval
import causetrace.retrieval as retrieval
from causetrace.cli import app
from causetrace.db import Connection
from causetrace.index import build_index
from causetrace.ingest import ingest
from causetrace.llm.fake import FakeProvider
from causetrace.split import load_split

runner = CliRunner()


@pytest.mark.parametrize(("command", "module"), [("retrieval", retrieval_eval),
                                                 ("causal", causal_eval)])  # fmt: skip
def test_eval_commands_pass_arguments_through(
    command: str, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        seen.append(list(argv or []))
        return 3

    monkeypatch.setattr(module, "main", fake_main)
    result = runner.invoke(app, ["eval", command, "--limit", "5", "--seed", "7"])
    assert result.exit_code == 3
    assert seen == [["--limit", "5", "--seed", "7"]]


@pytest.fixture(scope="module")
def indexed(db_url: str) -> Iterator[Connection]:
    with psycopg.connect(db_url, autocommit=True) as conn:
        reset(conn)
        ingest(conn, FIXTURES)
        build_index(conn, FakeEmbedder(), batch_size=256)
        yield conn


def test_ask_prints_the_grounded_answer(
    indexed: Connection, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = retrieval.Retriever

    def fake_retriever(conn: Connection) -> retrieval.Retriever:
        return real(conn, FakeEmbedder(), FakeReranker())

    monkeypatch.setattr(retrieval, "Retriever", fake_retriever)
    monkeypatch.setattr(agent_main, "make_provider", lambda choice: FakeProvider(Responder()))
    ev_id = load_split(indexed).held_out[0]
    result = runner.invoke(app, ["ask", ev_id, "--database-url", db_url])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["ev_id"] == ev_id and out["findings"]
    assert out["guardrail"]["grounding_score"] == 1.0
    missing = runner.invoke(app, ["ask", "no-such-event", "--database-url", db_url])
    assert missing.exit_code == 2
