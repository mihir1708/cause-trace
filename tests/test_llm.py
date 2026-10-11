"""LLM provider layer: costs, spend tracking, the Anthropic adapter (with a stub client)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from causetrace.config import max_usd
from causetrace.llm import BudgetExceededError, LLMResponseError, SpendTracker, cost_usd
from causetrace.llm.anthropic_provider import AnthropicProvider
from causetrace.llm.fake import FakeProvider
from causetrace.llm.provider import shared_tracker

SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}


def test_cost_usd() -> None:
    # Up to 100K prompt tokens: $0.10 in, $0.50 out per million.
    assert cost_usd("claude-haiku-5-5", 100_000, 1_000_000) == pytest.approx(0.01 + 0.50)
    # Above 100K prompt tokens the long-context prices apply.
    assert cost_usd("claude-haiku-5-5", 200_000, 1_000) == pytest.approx(0.1 + 0.0025)
    # Cache writes 1.25x, cache reads 0.1x the input price.
    assert cost_usd("claude-haiku-5-5", 1_000, 0, 2_000, 10_000) == pytest.approx(450 / 1e6)
    with pytest.raises(KeyError):
        cost_usd("unknown-model", 1, 1)


def test_tracker_raises_once_budget_is_reached() -> None:
    tracker = SpendTracker(max_usd=0.05)
    provider = FakeProvider(lambda s, p, sc: {"question": "q"}, tracker=tracker, usd_per_call=0.02)
    for _ in range(3):  # 0.00 -> 0.02 -> 0.04 -> 0.06
        provider.generate_json("sys", "prompt", SCHEMA)
    assert tracker.spent_usd == pytest.approx(0.06)
    with pytest.raises(BudgetExceededError):
        provider.generate_json("sys", "prompt", SCHEMA)
    assert len(tracker.calls) == 3


def test_shared_tracker_uses_max_usd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("MAX_USD", raising=False)
    assert max_usd() == 10.0
    monkeypatch.setenv("MAX_USD", "2.5")
    shared_tracker.cache_clear()
    try:
        assert shared_tracker().max_usd == 2.5
    finally:
        shared_tracker.cache_clear()


def test_dotenv_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MAX_USD", raising=False)
    (tmp_path / ".env").write_text("# comment\nMAX_USD='3'\n")
    assert max_usd() == 3.0


class StubMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _response(stop: str = "end_turn", text: str = '{"question": "Why?"}') -> Any:
    return SimpleNamespace(
        stop_reason=stop,
        stop_details=None,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(
            input_tokens=1_000,
            output_tokens=200,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=0,
        ),
    )


def _provider(
    response: Any, tracker: SpendTracker | None = None
) -> tuple[AnthropicProvider, StubMessages]:
    messages = StubMessages(response)
    client = SimpleNamespace(messages=messages)
    provider = AnthropicProvider(client=client, tracker=tracker or SpendTracker(max_usd=10))
    return provider, messages


def test_anthropic_provider_request_and_usage() -> None:
    provider, messages = _provider(_response())
    parsed, usage = provider.generate_json("system text", "the prompt", SCHEMA)
    assert parsed == {"question": "Why?"}
    assert messages.kwargs["model"] == "claude-haiku-5-5"
    assert messages.kwargs["system"] == "system text"
    assert messages.kwargs["messages"] == [{"role": "user", "content": "the prompt"}]
    assert messages.kwargs["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert (usage.input_tokens, usage.output_tokens) == (1_000, 200)
    assert usage.cost_usd == pytest.approx((1_000 * 0.10 + 200 * 0.50) / 1e6)
    assert usage.latency_s >= 0
    assert provider.tracker.spent_usd == pytest.approx(usage.cost_usd)


@pytest.mark.parametrize(
    ("stop", "text"),
    [
        ("refusal", "{}"),
        ("max_tokens", '{"question": "Wh'),
        ("end_turn", "not json"),
        ("end_turn", "[1]"),
    ],
)
def test_anthropic_provider_unusable_output(stop: str, text: str) -> None:
    provider, _ = _provider(_response(stop, text))
    with pytest.raises(LLMResponseError):
        provider.generate_json("s", "p", SCHEMA)
    assert len(provider.tracker.calls) == 1  # the call is still billed


def test_anthropic_provider_checks_budget_before_calling() -> None:
    tracker = SpendTracker(max_usd=0.0)
    provider, messages = _provider(_response(), tracker)
    with pytest.raises(BudgetExceededError):
        provider.generate_json("s", "p", SCHEMA)
    assert messages.kwargs == {}  # no request was sent
