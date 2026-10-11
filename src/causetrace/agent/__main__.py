"""Run the agent for one event and print the result JSON with its trace.

Usage: python -m causetrace.agent <ev_id> [--provider auto|anthropic|ollama]

``auto`` uses Claude (claude-haiku-5-5) when ANTHROPIC_API_KEY is set, with the local
Ollama model as a free fallback, and Ollama alone otherwise. Needs the ``ml`` dependency
group for retrieval models.
"""

import argparse
import json
import sys

from causetrace.agent.graph import Agent
from causetrace.config import anthropic_api_key
from causetrace.db import connect
from causetrace.llm import FallbackProvider, Provider
from causetrace.llm.ollama_provider import OllamaProvider
from causetrace.retrieval import Retriever


def make_provider(choice: str) -> Provider:
    if choice == "ollama" or (choice == "auto" and not anthropic_api_key()):
        return OllamaProvider()
    from causetrace.llm.anthropic_provider import AnthropicProvider

    if choice == "anthropic":
        return AnthropicProvider()
    return FallbackProvider(AnthropicProvider(), OllamaProvider())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CauseTrace agent for one event.")
    parser.add_argument("ev_id")
    parser.add_argument("--provider", choices=["auto", "anthropic", "ollama"], default="auto")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    with connect(args.database_url) as conn:
        agent = Agent(conn, Retriever(conn), make_provider(args.provider))
        try:
            result = agent.run(args.ev_id)
        except LookupError as err:
            print(str(err), file=sys.stderr)
            return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
