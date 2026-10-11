"""Leakage gate for the CONDITIONAL factual narrative (``narratives.narr_accp``).

See docs/DATA_LEAKAGE.md. Sentences of the factual narrative that share a run of
``NGRAM`` consecutive words with the event's GOLD_ONLY narratives, or contain a whole gold
sentence of ``MIN_SENTENCE_WORDS`` or more words, are dropped; if none survive, the
narrative is withheld.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

NGRAM = 12
MIN_SENTENCE_WORDS = 4

_WORD = re.compile(r"[a-z0-9]+")
_PARAGRAPH = re.compile(r"\r?\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class SanitizedNarrative:
    text: str | None  # None = withheld
    sentences_total: int
    sentences_removed: int


def words(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens."""
    return _WORD.findall(text.lower())


def _split_sentences(text: str) -> list[tuple[int, str]]:
    """``(paragraph index, sentence)`` pairs; paragraphs are separated by line breaks."""
    return [
        (p, s.strip())
        for p, paragraph in enumerate(_PARAGRAPH.split(text))
        for s in _SENTENCE.split(paragraph)
        if s.strip()
    ]


def _gold_phrases(gold_texts: Iterable[str | None], n: int) -> dict[int, set[tuple[str, ...]]]:
    """Word runs to block, keyed by run length.

    Every n-gram of each gold text; the whole text if it is shorter than n words; and
    every gold sentence shorter than n words that has at least MIN_SENTENCE_WORDS words.
    """
    phrases: dict[int, set[tuple[str, ...]]] = {}

    def block(toks: list[str]) -> None:
        phrases.setdefault(len(toks), set()).add(tuple(toks))

    for text in gold_texts:
        toks = words(text or "")
        if not toks:
            continue
        if len(toks) < n:
            block(toks)
            continue
        phrases.setdefault(n, set()).update(
            tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)
        )
        for _, sentence in _split_sentences(text or ""):
            sentence_toks = words(sentence)
            if MIN_SENTENCE_WORDS <= len(sentence_toks) < n:
                block(sentence_toks)
    return phrases


def sanitize_factual_narrative(
    narr_accp: str | None, gold_texts: Iterable[str | None], n: int = NGRAM
) -> SanitizedNarrative:
    """Drop sentences of ``narr_accp`` that share gold word runs (see module docstring)."""
    sentences = _split_sentences(narr_accp or "")
    phrases = _gold_phrases(gold_texts, n)

    kept = list(range(len(sentences)))
    while phrases:
        stream: list[str] = []
        owner: list[int] = []
        for idx in kept:
            toks = words(sentences[idx][1])
            stream += toks
            owner += [idx] * len(toks)
        hit: set[int] = set()
        for length, blocked in phrases.items():
            for i in range(len(stream) - length + 1):
                if tuple(stream[i : i + length]) in blocked:
                    hit.update(owner[i : i + length])
        if not hit:
            break
        kept = [idx for idx in kept if idx not in hit]

    paragraphs: dict[int, list[str]] = {}
    for idx in kept:
        p, sentence = sentences[idx]
        paragraphs.setdefault(p, []).append(sentence)
    text = "\n\n".join(" ".join(ss) for _, ss in sorted(paragraphs.items())) or None
    return SanitizedNarrative(
        text=text,
        sentences_total=len(sentences),
        sentences_removed=len(sentences) - len(kept),
    )
