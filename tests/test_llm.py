"""LLM provider layer: costs, spend tracking, the Anthropic adapter (with a stub client)."""

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from causetrace.config import LLM_TIMEOUT_S, max_usd
from causetrace.llm import (
    BudgetExceededError,
    FallbackProvider,
    LLMResponseError,
    ProviderUnavailableError,
    SpendTracker,
    cost_usd,
)
from causetrace.llm.anthropic_provider import AnthropicProvider
from causetrace.llm.fake import FakeProvider
from causetrace.llm.ollama_provider import OllamaProvider
from causetrace.llm.openai_provider import OpenAIProvider
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
    assert messages.kwargs["system"] == [{"type": "text", "text": "system text"}]
    assert "temperature" not in messages.kwargs and "top_p" not in messages.kwargs
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


def test_anthropic_provider_caches_system_blocks_and_sets_effort() -> None:
    provider, messages = _provider(_response())
    provider.generate_json(
        ["fixed prompt", "taxonomy"], "p", SCHEMA, effort="medium", cache_system=True
    )
    assert messages.kwargs["system"] == [
        {"type": "text", "text": "fixed prompt"},
        {"type": "text", "text": "taxonomy", "cache_control": {"type": "ephemeral"}},
    ]
    assert messages.kwargs["output_config"]["effort"] == "medium"


def test_missing_credentials_make_the_provider_unavailable() -> None:
    class NoCredentials:
        def create(self, **kwargs: Any) -> Any:
            raise TypeError("Could not resolve authentication method.")

    provider = AnthropicProvider(client=SimpleNamespace(messages=NoCredentials()),
                                 tracker=SpendTracker(max_usd=10))  # fmt: skip
    with pytest.raises(ProviderUnavailableError):
        provider.generate_json("s", "p", SCHEMA)


class Raising:
    """A fake client whose ``messages.create`` raises ``err``."""

    def __init__(self, err: Exception) -> None:
        self.err = err
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        raise self.err


def _request() -> Any:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    "err",
    [
        anthropic.APITimeoutError(request=_request()),
        anthropic.OverloadedError(
            "Overloaded",
            response=httpx2.Response(529, request=_request()),
            body={"type": "error", "error": {"type": "overloaded_error"}},
        ),
    ],
    ids=["timeout", "overloaded-529"],
)
def test_timeout_and_overload_make_the_provider_unavailable(err: Exception) -> None:
    client = Raising(err)
    provider = AnthropicProvider(
        client=SimpleNamespace(messages=client), tracker=SpendTracker(max_usd=10)
    )
    with pytest.raises(ProviderUnavailableError, match=type(err).__name__):
        provider.generate_json("s", "p", SCHEMA)
    assert client.calls == 1 and provider.tracker.calls == []  # nothing billed


def test_default_client_has_a_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # builds the client; no request sent
    provider = AnthropicProvider(tracker=SpendTracker(max_usd=10))
    assert provider.timeout == LLM_TIMEOUT_S == 120
    assert provider._client.timeout == 120 and provider._client.max_retries == 2
    assert AnthropicProvider(tracker=SpendTracker(max_usd=10), timeout=5)._client.timeout == 5


def test_fallback_provider_switches_once_on_unavailability() -> None:
    class Down:
        model = "down"
        tracker = SpendTracker(max_usd=10)

        def generate_json(self, *a: Any, **k: Any) -> Any:
            raise ProviderUnavailableError("no connection")

    backup = FakeProvider(lambda s, p, sc: {"question": "from fallback"})
    provider = FallbackProvider(Down(), backup)
    assert provider.generate_json("s", "p", SCHEMA)[0] == {"question": "from fallback"}
    assert provider.fell_back and provider.model == "fake-model"
    provider.generate_json("s", "p", SCHEMA, effort="low", cache_system=True)
    assert backup.calls[-1]["effort"] == "low" and backup.calls[-1]["cache_system"] is True


def test_fallback_provider_does_not_hide_budget_errors() -> None:
    primary = FakeProvider(lambda s, p, sc: {}, tracker=SpendTracker(max_usd=0))
    provider = FallbackProvider(primary, FakeProvider(lambda s, p, sc: {"question": "x"}))
    with pytest.raises(BudgetExceededError):
        provider.generate_json("s", "p", SCHEMA)


def test_ollama_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"message": {"content": '{"question": "local?"}'}, "prompt_eval_count": 40,
                 "eval_count": 7}
            ).encode()  # fmt: skip

    def fake_urlopen(request: Any, timeout: float) -> Response:
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    tracker = SpendTracker(max_usd=10)
    provider = OllamaProvider("llama3.1:8b", tracker=tracker)
    parsed, usage = provider.generate_json(["sys a", "sys b"], "prompt", SCHEMA, effort="medium")
    assert parsed == {"question": "local?"}
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["body"]["format"] == SCHEMA and sent["body"]["stream"] is False
    assert sent["body"]["messages"][0] == {"role": "system", "content": "sys a\n\nsys b"}
    assert "options" not in sent["body"]  # no sampling parameters
    assert (usage.input_tokens, usage.output_tokens, usage.cost_usd) == (40, 7, 0.0)
    assert len(tracker.calls) == 1
    with pytest.raises(BudgetExceededError):  # the spend guard runs even for free calls
        OllamaProvider(tracker=SpendTracker(max_usd=0)).generate_json("s", "p", SCHEMA)


def test_ollama_unreachable_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: Any, timeout: float) -> Any:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(ProviderUnavailableError):
        OllamaProvider(tracker=SpendTracker(max_usd=10)).generate_json("s", "p", SCHEMA)


def test_openai_provider_request_and_cost() -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1_000, completion_tokens=100),
            choices=[SimpleNamespace(finish_reason="stop",
                                     message=SimpleNamespace(content='{"question": "o?"}'))],
        )  # fmt: skip

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = OpenAIProvider("some-model", usd_per_mtok_input=1.0, usd_per_mtok_output=2.0,
                              client=client, tracker=SpendTracker(max_usd=10))  # fmt: skip
    parsed, usage = provider.generate_json("s", "p", SCHEMA)
    assert parsed == {"question": "o?"}
    assert captured["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "temperature" not in captured
    assert usage.cost_usd == pytest.approx((1_000 * 1.0 + 100 * 2.0) / 1e6)
