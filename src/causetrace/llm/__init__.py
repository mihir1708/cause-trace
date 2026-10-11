"""The single interface for LLM calls (CLAUDE.md rule 3).

Every call goes through a ``Provider``, which records tokens, USD cost and latency in a
``SpendTracker``; the shared tracker stops calls once ``MAX_USD`` is reached.
"""

from causetrace.llm.provider import (
    BudgetExceededError,
    Effort,
    FallbackProvider,
    LLMResponseError,
    Provider,
    ProviderUnavailableError,
    SpendTracker,
    SystemPrompt,
    Usage,
    cost_usd,
    shared_tracker,
)

__all__ = [
    "BudgetExceededError",
    "Effort",
    "FallbackProvider",
    "LLMResponseError",
    "Provider",
    "ProviderUnavailableError",
    "SpendTracker",
    "SystemPrompt",
    "Usage",
    "cost_usd",
    "shared_tracker",
]
