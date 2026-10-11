"""OpenAI chat models on the same interface (not used by default).

Prices are required arguments: this module has no price table, so spend is never guessed.
"""

import time
from typing import Any

from causetrace.llm.provider import (
    Effort,
    LLMResponseError,
    ProviderUnavailableError,
    SpendTracker,
    SystemPrompt,
    Usage,
    parse_json_object,
    shared_tracker,
    system_text,
)


class OpenAIProvider:
    """``chat.completions.create`` with a strict ``json_schema`` response format.

    No temperature or other sampling parameter is sent; effort and caching hints are
    ignored (OpenAI caches long prompt prefixes automatically).
    """

    def __init__(
        self,
        model: str,
        *,
        usd_per_mtok_input: float,
        usd_per_mtok_output: float,
        client: Any = None,
        tracker: SpendTracker | None = None,
    ) -> None:
        self.model = model
        self.usd_per_mtok_input = usd_per_mtok_input
        self.usd_per_mtok_output = usd_per_mtok_output
        self.tracker = tracker or shared_tracker()
        if client is None:
            import openai

            client = openai.OpenAI()  # OPENAI_API_KEY from the environment
        self._client: Any = client

    def generate_json(
        self,
        system: SystemPrompt,
        prompt: str,
        schema: dict[str, Any],
        *,
        effort: Effort | None = None,
        cache_system: bool = False,
    ) -> tuple[dict[str, Any], Usage]:
        import openai

        self.tracker.check()
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_text(system)},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": schema, "strict": True},
                },
            )
        except (
            openai.APIConnectionError,
            openai.AuthenticationError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as err:
            raise ProviderUnavailableError(f"{type(err).__name__}: {err}") from err
        u = response.usage
        input_tokens = int(getattr(u, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(u, "completion_tokens", 0) or 0)
        usage = Usage(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=(
                input_tokens * self.usd_per_mtok_input + output_tokens * self.usd_per_mtok_output
            )
            / 1_000_000,
            latency_s=time.perf_counter() - start,
        )
        self.tracker.record(usage)
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMResponseError("output truncated")
        return parse_json_object(getattr(choice.message, "content", None)), usage
