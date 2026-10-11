"""A deterministic provider for tests: no network, no cost unless a price is given."""

from collections.abc import Callable
from typing import Any

from causetrace.llm.provider import SpendTracker, Usage

Responder = Callable[[str, str, dict[str, Any]], dict[str, Any]]


class FakeProvider:
    """Answers with ``responder(system, prompt, schema)``; tokens = whitespace words.

    ``usd_per_call`` lets tests exercise the spend tracker.
    """

    def __init__(
        self,
        responder: Responder,
        *,
        tracker: SpendTracker | None = None,
        usd_per_call: float = 0.0,
        model: str = "fake-model",
    ) -> None:
        self.responder = responder
        self.tracker = tracker or SpendTracker(max_usd=float("inf"))
        self.usd_per_call = usd_per_call
        self.model = model
        self.prompts: list[str] = []

    def generate_json(
        self, system: str, prompt: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], Usage]:
        self.tracker.check()
        self.prompts.append(prompt)
        parsed = self.responder(system, prompt, schema)
        usage = Usage(
            model=self.model,
            input_tokens=len(system.split()) + len(prompt.split()),
            output_tokens=sum(len(str(v).split()) for v in parsed.values()),
            cost_usd=self.usd_per_call,
            latency_s=0.0,
        )
        self.tracker.record(usage)
        return parsed, usage
