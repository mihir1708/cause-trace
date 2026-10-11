"""Input and output guardrails for the served agent.

Input: free-text packets are capped in length and rejected when they look like a prompt
injection (an instruction to the model rather than a description of an incident). Output:
a finding is kept only with at least one valid citation (a passage retrieved in this run,
from another event, dated before the run's cutoff); the grounding score is the share of
findings kept.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

MAX_NARRATIVE_CHARS = 20_000
MAX_FIELD_CHARS = 200

# Phrases that address the model instead of describing an accident. Kept narrow on purpose:
# NTSB narratives quote ATC ("disregard previous instructions", "you are now number two")
# and crew roles ("acting as safety pilot"), so those words alone are not flagged; a test
# checks that no fixture narrative trips a pattern.
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"\b(ignore|forget)\s+(all\s+|any\s+)?(of\s+)?(the\s+|your\s+)?"
        r"(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|rules)",
        r"\b(system|developer)\s+prompt\b",
        r"</?\s*(system|assistant|user|instructions?)\s*>",
        r"^\s*#{1,6}\s*(system|instructions?|assistant)\b",
        r"\byou\s+are\s+(an?\s+)?(ai|assistant|language\s+model|llm|chatgpt|claude)\b",
        r"\bas\s+an?\s+(ai|language\s+model)\b",
        r"\b(reveal|print|repeat)\s+(your|the)\s+(system\s+)?(prompt|instructions)\b",
        r"\b(jailbreak|do\s+anything\s+now)\b",
        r"\b(output|return|respond\s+with|predict)\s+(only\s+)?(the\s+)?"
        r"(labels?|findings?|codes?)\s*:?\s*\d{2}-\d{2}-\d{2}",
    )
)


class GuardrailViolation(ValueError):
    """A request rejected by an input guardrail (HTTP 400)."""


def check_free_text(fields: Mapping[str, str | None]) -> None:
    """Reject over-long fields and prompt-injection patterns; ``narrative`` gets the big cap."""
    for name, text in fields.items():
        if not text:
            continue
        cap = MAX_NARRATIVE_CHARS if name == "narrative" else MAX_FIELD_CHARS
        if len(text) > cap:
            raise GuardrailViolation(f"{name} is longer than {cap} characters")
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                raise GuardrailViolation(
                    f"{name} looks like an instruction to the model, not incident facts"
                )


def ground_findings(
    findings: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    *,
    before_date: str,
    ev_id: str,
) -> tuple[list[dict[str, Any]], int, float | None]:
    """Keep findings with at least one valid citation; return (kept, dropped, score).

    Citations are filtered to the valid ones. ``before_date`` is the run's cutoff (ISO
    date). The score is kept / all findings, None when there were no findings.
    """
    allowed = {
        str(p["passage_id"])
        for p in passages
        if str(p["event_id"]) != ev_id and str(p["event_date"]) < before_date
    }
    kept: list[dict[str, Any]] = []
    for f in findings:
        cites = [c for c in dict.fromkeys(f.get("citations") or []) if c in allowed]
        if cites:
            kept.append({**f, "citations": cites})
    dropped = len(findings) - len(kept)
    score = len(kept) / len(findings) if findings else None
    return kept, dropped, score
