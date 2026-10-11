"""Local models through Ollama (free): the fallback when Claude is unavailable."""

import json
import time
import urllib.error
import urllib.request
from typing import Any

from causetrace.config import OLLAMA_MODEL, OLLAMA_URL
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


class OllamaProvider:
    """``POST /api/chat`` with ``format`` set to the JSON schema; cost is always $0.

    The spend guard still runs before every call. Effort and caching do not apply; no
    sampling options are sent, so the model's defaults are used.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        *,
        url: str = OLLAMA_URL,
        tracker: SpendTracker | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.tracker = tracker or shared_tracker()
        self.timeout_s = timeout_s

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
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text(system)},
                {"role": "user", "content": prompt},
            ],
            "format": schema,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            raise ProviderUnavailableError(f"Ollama at {self.url}: {err}") from err
        usage = Usage(
            model=self.model,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            cost_usd=0.0,
            latency_s=time.perf_counter() - start,
        )
        self.tracker.record(usage)
        message = data.get("message") or {}
        if not isinstance(message, dict):
            raise LLMResponseError("unexpected Ollama response")
        return parse_json_object(message.get("content")), usage
