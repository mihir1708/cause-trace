"""Claude via the official ``anthropic`` SDK, with structured outputs for JSON."""

import time
from typing import Any

import anthropic
from anthropic.types import MessageParam, OutputConfigParam, TextBlockParam

from causetrace.config import LLM_MODEL, anthropic_api_key
from causetrace.llm.provider import (
    Effort,
    LLMResponseError,
    ProviderUnavailableError,
    SpendTracker,
    SystemPrompt,
    Usage,
    cost_usd,
    parse_json_object,
    shared_tracker,
)

MAX_TOKENS = 16_000


class AnthropicProvider:
    """``generate_json`` through ``client.messages.create`` with ``output_config.format``.

    ``client`` can be injected (tests pass a stub); by default it is built from
    ANTHROPIC_API_KEY (environment or .env). With ``cache_system`` the last system block
    carries ``cache_control``, so the fixed system prompt and taxonomy are cached. No
    temperature or other sampling parameter is sent.
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
        self.effort: Effort = effort
        self.max_tokens = max_tokens
        self.tracker = tracker or shared_tracker()
        self._client: Any = (
            client if client is not None else anthropic.Anthropic(api_key=anthropic_api_key())
        )

    def generate_json(
        self,
        system: SystemPrompt,
        prompt: str,
        schema: dict[str, Any],
        *,
        effort: Effort | None = None,
        cache_system: bool = False,
    ) -> tuple[dict[str, Any], Usage]:
        self.tracker.check()
        blocks = [system] if isinstance(system, str) else list(system)
        system_param: list[TextBlockParam] = [{"type": "text", "text": b} for b in blocks]
        if cache_system and system_param:
            system_param[-1]["cache_control"] = {"type": "ephemeral"}
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        output_config: OutputConfigParam = {
            "effort": effort or self.effort,
            "format": {"type": "json_schema", "schema": schema},
        }
        start = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_param,
                messages=messages,
                output_config=output_config,
            )
        except (
            anthropic.APIConnectionError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            TypeError,  # the SDK raises TypeError when no credentials can be resolved
        ) as err:
            raise ProviderUnavailableError(f"{type(err).__name__}: {err}") from err
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
        return parse_json_object(text), usage
