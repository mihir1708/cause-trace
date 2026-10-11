"""Provider interface, usage accounting and the spend tracker."""

import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from causetrace.config import max_usd

log = logging.getLogger(__name__)

# USD per million tokens: (input, output) for prompts up to the threshold, then above it.
# Cache writes cost 1.25x input, cache reads 0.1x input.
LONG_CONTEXT_THRESHOLD = 100_000
PRICES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "claude-haiku-5-5": ((0.10, 0.50), (0.50, 2.50)),
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


class BudgetExceededError(RuntimeError):
    """Raised before a call once the tracker's spend has reached its limit."""


class LLMResponseError(RuntimeError):
    """The model returned no usable JSON (refusal, truncation, invalid output)."""


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
    """Returns JSON matching ``schema`` (a JSON Schema object) and the call's usage."""

    tracker: SpendTracker

    def generate_json(
        self, system: str, prompt: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], Usage]: ...
