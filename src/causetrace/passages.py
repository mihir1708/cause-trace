"""Build retrieval passages from the ingested narratives.

Per event, one passage per section: ``factual`` (narr_accp) and ``analysis`` (narr_accf),
both chunked when long, and ``probable_cause`` (narr_cause). Narratives of all aircraft
of an event are joined (identical texts once), and NTSB section headings glued to the next
word ("HISTORY OF FLIGHTOn") are split. Passages contain GOLD_ONLY text of their
own event; search must filter by date and target event (docs/DATA_LEAKAGE.md).
"""

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date

from causetrace.db import Connection

CHUNK_WORDS = 300
CHUNK_OVERLAP = 50
MAX_SINGLE_CHUNK_WORDS = 400

SECTIONS = (("factual", "narr_accp"), ("analysis", "narr_accf"), ("probable_cause", "narr_cause"))
CHUNKED_SECTIONS = frozenset({"factual", "analysis"})  # can exceed the 512-token embedder window

# Section headings of NTSB factual narratives. The export often drops the line break after
# them, gluing the heading to the next word ("HISTORY OF FLIGHTOn", "AIRPORT INFORMATIONhe").
NTSB_HEADINGS = (
    "HISTORY OF FLIGHT",
    "PERSONNEL INFORMATION",
    "PILOT INFORMATION",
    "AIRCRAFT INFORMATION",
    "METEOROLOGICAL INFORMATION",
    "AIRPORT INFORMATION",
    "AIDS TO NAVIGATION",
    "COMMUNICATIONS",
    "FLIGHT RECORDERS",
    "WRECKAGE AND IMPACT INFORMATION",
    "MEDICAL AND PATHOLOGICAL INFORMATION",
    "FIRE",
    "SURVIVAL ASPECTS",
    "TESTS AND RESEARCH",
    "ORGANIZATIONAL AND MANAGEMENT INFORMATION",
    "ADDITIONAL INFORMATION",
    "INJURIES TO PERSONS",
    "DAMAGE TO AIRCRAFT",
    "OTHER DAMAGE",
)
# "FIRE" begins real uppercase words (FIREFIGHTING, FIREWALL), so it splits only before a
# capitalized word or lowercase letters; the other headings split before any letter or digit
# ("PERSONNEL INFORMATIONA review", "AIRPORT INFORMATIONN93AA").
_PREFIX_HEADINGS = frozenset({"FIRE"})


def _heading_pattern(headings: list[str], lookahead: str) -> str:
    alternatives = sorted(headings, key=len, reverse=True)
    return (
        r"(?<![A-Za-z])("
        + "|".join(r"\s+".join(map(re.escape, h.split())) for h in alternatives)
        + r")"
        + lookahead
    )


_GLUED_HEADING = re.compile(
    _heading_pattern([h for h in NTSB_HEADINGS if h not in _PREFIX_HEADINGS], r"(?=[A-Za-z0-9])")
    + "|"
    + _heading_pattern(sorted(_PREFIX_HEADINGS), r"(?=[A-Z][a-z]|[a-z])")
)


def split_glued_headings(text: str) -> str:
    """Insert a space after an NTSB heading glued to the next word."""
    return _GLUED_HEADING.sub(lambda m: (m.group(1) or m.group(2)) + " ", text)


@dataclass(frozen=True)
class Passage:
    passage_id: str
    event_id: str
    event_date: date
    section_type: str
    chunk_index: int
    aircraft_category: str | None
    phase_of_flight: str | None
    content: str

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def chunk_words(
    text: str,
    size: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
    max_single: int = MAX_SINGLE_CHUNK_WORDS,
) -> list[str]:
    """Split ``text`` into chunks of about ``size`` words overlapping by ``overlap`` words.

    Texts of at most ``max_single`` words stay whole. Longer texts get
    ``ceil((n - overlap) / (size - overlap))`` chunks of equal length (<= size), so there
    is no short tail chunk; consecutive chunks share exactly ``overlap`` words.
    """
    toks = text.split()
    if not toks:
        return []
    if len(toks) <= max_single:
        return [text.strip()]
    n_chunks = math.ceil((len(toks) - overlap) / (size - overlap))
    stride = (len(toks) - overlap) / n_chunks
    chunks = []
    for i in range(n_chunks):
        start = round(i * stride)
        end = len(toks) if i == n_chunks - 1 else round((i + 1) * stride) + overlap
        chunks.append(" ".join(toks[start:end]))
    return chunks


def _join_distinct(texts: list[str | None]) -> str:
    seen: list[str] = []
    for t in texts:
        t = (t or "").strip()
        if t and t not in seen:
            seen.append(t)
    return "\n\n".join(seen)


_SOURCE_SQL = """
SELECT e.ev_id, e.ev_date,
       first_a.acft_category, first_a.phase_flt_spec,
       coalesce(array_agg(n.narr_accp ORDER BY n.aircraft_key)
                FILTER (WHERE n.ev_id IS NOT NULL), '{}'),
       coalesce(array_agg(n.narr_accf ORDER BY n.aircraft_key)
                FILTER (WHERE n.ev_id IS NOT NULL), '{}'),
       coalesce(array_agg(n.narr_cause ORDER BY n.aircraft_key)
                FILTER (WHERE n.ev_id IS NOT NULL), '{}')
FROM events e
LEFT JOIN LATERAL (
    SELECT acft_category, phase_flt_spec FROM aircraft a
    WHERE a.ev_id = e.ev_id ORDER BY a.aircraft_key LIMIT 1
) first_a ON true
LEFT JOIN narratives n ON n.ev_id = e.ev_id
GROUP BY e.ev_id, e.ev_date, first_a.acft_category, first_a.phase_flt_spec
ORDER BY e.ev_id
"""


def generate_passages(conn: Connection) -> list[Passage]:
    """All passages for every event in the database, in a stable order."""
    passages: list[Passage] = []
    for ev_id, ev_date, category, phase, accp, accf, cause in conn.execute(_SOURCE_SQL):
        texts = {"factual": accp, "analysis": accf, "probable_cause": cause}
        for section, _ in SECTIONS:
            content = split_glued_headings(_join_distinct(texts[section]))
            if section in CHUNKED_SECTIONS:
                chunks = chunk_words(content)
            else:
                chunks = [content] if content else []
            for i, chunk in enumerate(chunks):
                passages.append(
                    Passage(
                        passage_id=f"{ev_id}:{section}:{i}",
                        event_id=ev_id,
                        event_date=ev_date,
                        section_type=section,
                        chunk_index=i,
                        aircraft_category=category,
                        phase_of_flight=phase,
                        content=chunk,
                    )
                )
    return passages


_UPSERT_SQL = """
INSERT INTO passages (passage_id, event_id, event_date, section_type, chunk_index,
                      aircraft_category, phase_of_flight, content, content_sha256)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (passage_id) DO UPDATE SET
    event_id = EXCLUDED.event_id,
    event_date = EXCLUDED.event_date,
    section_type = EXCLUDED.section_type,
    chunk_index = EXCLUDED.chunk_index,
    aircraft_category = EXCLUDED.aircraft_category,
    phase_of_flight = EXCLUDED.phase_of_flight,
    content = EXCLUDED.content,
    content_sha256 = EXCLUDED.content_sha256,
    embedding = CASE WHEN passages.content_sha256 = EXCLUDED.content_sha256
                     THEN passages.embedding END,
    embedding_model = CASE WHEN passages.content_sha256 = EXCLUDED.content_sha256
                           THEN passages.embedding_model END,
    embedding_revision = CASE WHEN passages.content_sha256 = EXCLUDED.content_sha256
                              THEN passages.embedding_revision END
"""


def build_passages(conn: Connection) -> dict[str, int]:
    """Upsert all passages and delete stale ones; return passage counts per section.

    Embeddings of passages whose text did not change are kept.
    """
    passages = generate_passages(conn)
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            _UPSERT_SQL,
            [
                (
                    p.passage_id,
                    p.event_id,
                    p.event_date,
                    p.section_type,
                    p.chunk_index,
                    p.aircraft_category,
                    p.phase_of_flight,
                    p.content,
                    p.content_sha256,
                )
                for p in passages
            ],
        )
        cur.execute(
            "DELETE FROM passages WHERE passage_id <> ALL(%s)",
            ([p.passage_id for p in passages],),
        )
    conn.commit()
    counts: dict[str, int] = {section: 0 for section, _ in SECTIONS}
    for p in passages:
        counts[p.section_type] += 1
    return counts
