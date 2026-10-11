"""Provider interface, usage accounting and the spend tracker."""

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Protocol

from causetrace.config import max_usd

log = logging.getLogger(__name__)

# USD per million tokens: (input, output) for prompts up to the threshold, then above it.
# Cache writes cost 1.25x input, cache reads 0.1x input.
LONG_CONTEXT_THRESHOLD = 100_000
# Source: Anthropic model pricing table (claude-api reference, cached 2026-10-06).
PRICES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "claude-haiku-5-5": ((0.10, 0.50), (0.50, 2.50)),
    "claude-sonnet-5-5": ((2.00, 10.00), (2.00, 10.00)),
    "claude-opus-5-5": ((4.00, 20.00), (4.00, 20.00)),
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


Effort = Literal["low", "medium", "high", "xhigh", "max"]
# A system prompt is one string or several blocks; with ``cache_system`` the provider marks
# the end of the blocks as a cache breakpoint (providers without caching join them).
SystemPrompt = str | Sequence[str]


def system_text(system: SystemPrompt) -> str:
    """The system prompt as one string (blocks joined by blank lines)."""
    return system if isinstance(system, str) else "\n\n".join(system)


class BudgetExceededError(RuntimeError):
    """Raised before a call once the tracker's spend has reached its limit."""


class LLMResponseError(RuntimeError):
    """The model returned no usable JSON (refusal, truncation, invalid output)."""


class ProviderUnavailableError(RuntimeError):
    """The provider cannot be reached or used (no credentials, connection, overload)."""


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """Cost of one call; unknown models are an error so spend is never under-counted."""
    if model not in PRICES:
        raise KeyError(f"no price for model {model!r}; add it to causetrace.llm.provider.PRICES")
    short, long = PRICES[model]
    prompt = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
    price_in, price_out = long if prompt > LONG_CONTEXT_THRESHOLD else short
    return (
        input_tokens * price_in
        + cache_creation_input_tokens * price_in * CACHE_WRITE_MULTIPLIER
        + cache_read_input_tokens * price_in * CACHE_READ_MULTIPLIER
        + output_tokens * price_out
    ) / 1_000_000


@dataclass
class SpendTracker:
    """Accumulates usage across providers; ``check`` raises once ``max_usd`` is reached."""

    max_usd: float
    spent_usd: float = 0.0
    calls: list[Usage] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        with self._lock:
            if self.spent_usd >= self.max_usd:
                raise BudgetExceededError(
                    f"LLM spend ${self.spent_usd:.4f} reached MAX_USD ${self.max_usd:.2f}"
                )

    def record(self, usage: Usage) -> None:
        with self._lock:
            self.spent_usd += usage.cost_usd
            self.calls.append(usage)
            total = self.spent_usd
        log.info(
            "llm call model=%s in=%d out=%d cost=$%.6f latency=%.2fs total=$%.4f",
            usage.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cost_usd,
            usage.latency_s,
            total,
        )

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.calls)


@lru_cache(maxsize=1)
def shared_tracker() -> SpendTracker:
    """The process-wide tracker, limited by MAX_USD (default 10)."""
    return SpendTracker(max_usd=max_usd())


class Provider(Protocol):
    """Returns JSON matching ``schema`` (a JSON Schema object) and the call's usage.

    ``effort`` sets the provider's reasoning effort when it has one (None: its default).
    ``cache_system`` asks the provider to cache the system blocks (ignored without caching).
    Providers never send temperature or other sampling parameters.
    """

    tracker: SpendTracker
    model: str

    def generate_json(
        self,
        system: SystemPrompt,
        prompt: str,
        schema: dict[str, Any],
        *,
        effort: Effort | None = None,
        cache_system: bool = False,
    ) -> tuple[dict[str, Any], Usage]: ...


def parse_json_object(text: str | None) -> dict[str, Any]:
    """Parse model output as a JSON object or raise ``LLMResponseError``."""
    import json

    if text is None:
        raise LLMResponseError("response has no text")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        raise LLMResponseError(f"invalid JSON from model: {err}") from err
    if not isinstance(parsed, dict):
        raise LLMResponseError("expected a JSON object")
    return parsed


class FallbackProvider:
    """Uses ``primary``; when it is unavailable (``ProviderUnavailableError``) uses ``fallback``.

    Budget errors and unusable output are not unavailability and propagate unchanged.
    """

    def __init__(self, primary: Provider, fallback: Provider) -> None:
        self.primary = primary
        self.fallback = fallback
        self.tracker = primary.tracker
        self.model = primary.model
        self.fell_back = False

    def generate_json(
        self,
        system: SystemPrompt,
        prompt: str,
        schema: dict[str, Any],
        *,
        effort: Effort | None = None,
        cache_system: bool = False,
    ) -> tuple[dict[str, Any], Usage]:
        if not self.fell_back:
            try:
                return self.primary.generate_json(
                    system, prompt, schema, effort=effort, cache_system=cache_system
                )
            except ProviderUnavailableError as err:
                log.warning("primary LLM unavailable (%s); using %s", err, self.fallback.model)
                self.fell_back = True
                self.model = self.fallback.model
        return self.fallback.generate_json(
            system, prompt, schema, effort=effort, cache_system=cache_system
        )
