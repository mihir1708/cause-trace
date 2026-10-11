"""Claude via the official ``anthropic`` SDK, with structured outputs for JSON."""

import json
import time
from typing import Any, Literal

import anthropic
from anthropic.types import MessageParam, OutputConfigParam

from causetrace.config import LLM_MODEL, anthropic_api_key
from causetrace.llm.provider import LLMResponseError, SpendTracker, Usage, cost_usd, shared_tracker

MAX_TOKENS = 16_000
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AnthropicProvider:
    """``generate_json`` through ``client.messages.create`` with ``output_config.format``.

    ``client`` can be injected (tests pass a stub); by default it is built from
    ANTHROPIC_API_KEY (environment or .env).
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        *,
        client: Any = None,
        tracker: SpendTracker | None = None,
        effort: Effort = "low",
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.tracker = tracker or shared_tracker()
        self._client = (
            client if client is not None else anthropic.Anthropic(api_key=anthropic_api_key())
        )

    def generate_json(
        self, system: str, prompt: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], Usage]:
        self.tracker.check()
        start = time.perf_counter()
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        output_config: OutputConfigParam = {
            "effort": self.effort,
            "format": {"type": "json_schema", "schema": schema},
        }
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config=output_config,
        )
        latency = time.perf_counter() - start
        u = response.usage
        cache_write = getattr(u, "cache_creation_input_tokens", None) or 0
        cache_read = getattr(u, "cache_read_input_tokens", None) or 0
        usage = Usage(
            model=self.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cost_usd=cost_usd(self.model, u.input_tokens, u.output_tokens, cache_write, cache_read),
            latency_s=latency,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        )
        self.tracker.record(usage)  # billed even when the output is unusable

        if response.stop_reason == "refusal":
            raise LLMResponseError(f"model refused: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise LLMResponseError("output truncated at max_tokens")
        text = next((getattr(b, "text", None) for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMResponseError("response has no text block")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as err:
            raise LLMResponseError(f"invalid JSON from model: {err}") from err
        if not isinstance(parsed, dict):
            raise LLMResponseError("expected a JSON object")
        return parsed, usage
