"""Known-item questions: sample history passages, draft a question per passage with the
LLM, and review the drafts by hand (keep, edit or drop)."""

import json
import random
import textwrap
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from causetrace.db import Connection
from causetrace.llm import LLMResponseError, Provider
from causetrace.split import HELD_OUT_FROM

DEFAULT_N = 220
DEFAULT_SEED = 42
SECTIONS = ("factual", "analysis", "probable_cause")

# Passages that cannot identify a single event are never sampled: very short ones
# ("None.", markup), and NTSB form text shared by many records (foreign investigations,
# FBI-led events, pointers to separate reports).
MIN_WORDS = 20
BOILERPLATE = (
    "for information purposes only",
    "did not determine the probable cause",
    "full report is available at",
)
_ELIGIBLE_SQL = f"""
WITH history AS (
    SELECT passage_id, event_id, section_type, event_date, content,
           lower(regexp_replace(btrim(content), '\\s+', ' ', 'g')) AS norm
    FROM passages
    WHERE event_date < %(cutoff)s
),
shared AS (
    SELECT norm FROM history GROUP BY norm HAVING count(DISTINCT event_id) > 1
)
SELECT passage_id, event_id, section_type, extract(year FROM event_date)::int, content
FROM history
WHERE array_length(regexp_split_to_array(btrim(content), '\\s+'), 1) >= %(min_words)s
  AND norm NOT IN (SELECT norm FROM shared)
  AND {" AND ".join(f"content NOT ILIKE '%%{marker}%%'" for marker in BOILERPLATE)}
ORDER BY passage_id
"""


@dataclass(frozen=True)
class SampledPassage:
    passage_id: str
    event_id: str
    section_type: str
    year: int
    content: str


def _largest_remainder(sizes: dict[int, int], n: int) -> dict[int, int]:
    """Split n across keys proportionally to sizes (largest remainder, ties by key)."""
    total = sum(sizes.values())
    if total == 0 or n == 0:
        return {key: 0 for key in sizes}
    n = min(n, total)
    quotas = {key: n * size / total for key, size in sizes.items()}
    alloc = {key: int(q) for key, q in quotas.items()}
    for key in sorted(sizes, key=lambda key: (alloc[key] - quotas[key], key))[
        : n - sum(alloc.values())
    ]:
        alloc[key] += 1
    return alloc


def sample_passages(
    conn: Connection, n: int = DEFAULT_N, seed: int = DEFAULT_SEED
) -> list[SampledPassage]:
    """``n`` eligible passages from events before HELD_OUT_FROM, stratified by section and year.

    Eligible: at least MIN_WORDS words, text not shared with any other event (whitespace and
    case ignored) and no BOILERPLATE marker, so each question has exactly one right event.

    The n is split evenly across section types (remainder to the earlier ones in
    SECTIONS); within a section it is split across event years proportionally to the
    passage count (largest remainder), then sampled with ``random.Random(seed)``.
    """
    rows = conn.execute(_ELIGIBLE_SQL, {"cutoff": HELD_OUT_FROM, "min_words": MIN_WORDS}).fetchall()
    cells: dict[str, dict[int, list[SampledPassage]]] = defaultdict(lambda: defaultdict(list))
    for pid, ev, section, year, content in rows:
        cells[section][year].append(SampledPassage(pid, ev, section, year, content))

    rng = random.Random(seed)
    per_section = {s: n // len(SECTIONS) + (i < n % len(SECTIONS)) for i, s in enumerate(SECTIONS)}
    picked: list[SampledPassage] = []
    for section in SECTIONS:
        by_year = cells.get(section, {})
        alloc = _largest_remainder({y: len(ps) for y, ps in by_year.items()}, per_section[section])
        for year in sorted(by_year):
            picked += rng.sample(by_year[year], alloc[year])
    return picked


SYSTEM_PROMPT = """You write evaluation questions for a search engine over NTSB aviation \
accident reports. You are shown one passage from one report. Write one question that an \
accident investigator might ask and that this passage answers.

Rules:
- The question must be answerable from the passage alone.
- Paraphrase. Do not copy distinctive phrases, and do not include registration numbers, \
names, dates, times, airport identifiers or exact place names from the passage.
- Ask about what happened or why (conditions, failures, decisions, causes), the way an \
investigator researching similar accidents would.
- One sentence, at most 30 words."""

QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}


def draft_question(provider: Provider, passage: SampledPassage) -> tuple[str, dict[str, Any]]:
    """One question for ``passage`` and the call's usage as a dict."""
    prompt = f"Section: {passage.section_type.replace('_', ' ')}\n\nPassage:\n{passage.content}"
    parsed, usage = provider.generate_json(SYSTEM_PROMPT, prompt, QUESTION_SCHEMA)
    question = str(parsed.get("question", "")).strip()
    if not question:
        raise LLMResponseError("empty question")
    return question, {
        "model": usage.model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": usage.cost_usd,
        "latency_s": round(usage.latency_s, 3),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record and flush, so an interrupted run loses nothing already written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def draft_all(
    provider: Provider,
    passages: list[SampledPassage],
    out: Path,
    log: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Draft questions for passages not yet in ``out``; return (written, skipped_failures).

    Resumable: passages already in ``out`` are skipped. Budget errors propagate.
    """
    done = {r["passage_id"] for r in read_jsonl(out)}
    written = failed = 0
    for p in passages:
        if p.passage_id in done:
            continue
        try:
            question, usage = draft_question(provider, p)
        except LLMResponseError as err:
            log(f"skip {p.passage_id}: {err}")
            failed += 1
            continue
        append_jsonl(
            out,
            {
                "passage_id": p.passage_id,
                "event_id": p.event_id,
                "section_type": p.section_type,
                "year": p.year,
                "question": question,
                "usage": usage,
            },
        )
        written += 1
    return written, failed


# --- Review ---------------------------------------------------------------------

DECISIONS = {"k": "keep", "e": "edit", "d": "drop"}


def pending_reviews(
    drafts: list[dict[str, Any]], reviewed: list[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    done = {r["passage_id"] for r in reviewed}
    return (d for d in drafts if d["passage_id"] not in done)


def review_loop(
    drafts: list[dict[str, Any]],
    reviewed_path: Path,
    passage_text: Callable[[str], str | None],
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    width: int = 100,
) -> int:
    """Show each unreviewed draft beside its passage and record keep/edit/drop.

    Each decision is appended to ``reviewed_path`` immediately; ``q`` (or EOF) stops and
    the next run resumes where this one stopped. Returns the number recorded now.
    """
    reviewed = read_jsonl(reviewed_path)
    todo = list(pending_reviews(drafts, reviewed))
    recorded = 0
    for i, draft in enumerate(todo, start=1):
        text = passage_text(draft["passage_id"]) or "(passage not found in the database)"
        show("=" * width)
        show(
            f"[{len(reviewed) + recorded + 1}/{len(drafts)}, {len(todo) - i} left after this]"
            f"  {draft['passage_id']}  ({draft.get('section_type', '?')})"
        )
        show("-" * width)
        show(textwrap.fill(text, width=width))
        show("-" * width)
        show(f"QUESTION: {draft['question']}")
        while True:
            try:
                answer = ask("[k]eep  [e]dit  [d]rop  [q]uit > ").strip().lower()
            except EOFError:
                answer = "q"
            if answer == "q":
                return recorded
            if answer in DECISIONS:
                break
        decision = DECISIONS[answer]
        question: str | None = draft["question"]
        if decision == "edit":
            new = ""
            while not new:
                try:
                    new = ask("new question > ").strip()
                except EOFError:
                    return recorded
            question = new
        elif decision == "drop":
            question = None
        append_jsonl(
            reviewed_path,
            {
                "passage_id": draft["passage_id"],
                "event_id": draft["event_id"],
                "section_type": draft.get("section_type"),
                "draft_question": draft["question"],
                "decision": decision,
                "question": question,
                "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        recorded += 1
    return recorded


def kept_questions(reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reviewed records whose decision is keep or edit (the last decision per passage wins)."""
    latest: dict[str, dict[str, Any]] = {}
    for r in reviewed:
        latest[r["passage_id"]] = r
    return [r for r in latest.values() if r["decision"] in ("keep", "edit") and r["question"]]
