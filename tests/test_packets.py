"""Leakage tests: packets carry SAFE fields and the gated narrative, never gold."""

import ast
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import FixtureData

from causetrace import packets
from causetrace.db import Connection
from causetrace.packets import get_incident_packet

NGRAM = 12
_WORD = re.compile(r"[a-z0-9]+")


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _WORD.findall(text.lower())
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def test_packet_for_every_fixture_event_has_no_gold_text_or_codes(
    loaded: tuple[Connection, dict[str, Any]], fixture_data: FixtureData
) -> None:
    conn, _ = loaded
    assert len(fixture_data.ev_ids) == 300
    for ev_id in fixture_data.ev_ids:
        dumped = get_incident_packet(ev_id, conn).model_dump(mode="json")
        text = "\n".join(_strings(dumped))
        blob = json.dumps(dumped)
        packet_norm = _norm(text)
        packet_grams = _ngrams(text, NGRAM)

        gold_texts = fixture_data.gold_texts(ev_id)
        assert gold_texts, ev_id
        for gold in gold_texts:
            assert _norm(gold) not in packet_norm, f"{ev_id}: whole gold text in packet"
            for sentence in re.split(r"(?<=[.!?])\s+", gold):
                if len(sentence.strip()) >= 40:
                    assert _norm(sentence) not in packet_norm, f"{ev_id}: {sentence!r}"
            shared = packet_grams & _ngrams(gold, NGRAM)
            assert not shared, f"{ev_id}: shares {NGRAM}-word run {' '.join(next(iter(shared)))}"

        for code in fixture_data.label_codes(ev_id) | fixture_data.finding_codes(ev_id):
            assert code not in blob, f"{ev_id}: code {code} in packet"


def test_packets_keep_most_narratives(
    loaded: tuple[Connection, dict[str, Any]], fixture_data: FixtureData
) -> None:
    """The gate must not silently drop everything."""
    conn, report = loaded
    with_narrative = sum(
        any(a.narr_accp for a in get_incident_packet(ev_id, conn).aircraft)
        for ev_id in fixture_data.ev_ids
    )
    gate = report["narrative_sanitization"]
    assert gate["narratives"] == 303
    assert with_narrative >= 200
    assert gate["withheld"] < gate["narratives"]


def test_identical_factual_and_analysis_narrative_is_withheld(
    loaded: tuple[Connection, dict[str, Any]], fixture_data: FixtureData
) -> None:
    conn, _ = loaded
    identical = [
        (n["ev_id"], int(n["Aircraft_Key"]))
        for n in fixture_data.narratives
        if n["narr_accp"].strip() == n["narr_accf"].strip()
    ]
    assert identical
    for ev_id, key in identical:
        aircraft = {a.aircraft_key: a for a in get_incident_packet(ev_id, conn).aircraft}
        assert aircraft[key].narr_accp is None


def test_packet_shape(loaded: tuple[Connection, dict[str, Any]]) -> None:
    conn, _ = loaded
    packet = get_incident_packet("20080219X00204", conn)
    assert set(packet.model_dump()) == {
        "ev_id",
        "ev_date",
        "ev_time",
        "ev_timezone",
        "location",
        "weather",
        "injuries",
        "aircraft",
    }
    assert packet.ev_date.isoformat() == "2008-01-11"
    assert packet.aircraft[0].make == "Cessna"
    assert packet.aircraft[0].damage == "SUBS"
    assert packet.weather.basic_conditions == "VMC"


def test_unknown_event_raises(loaded: tuple[Connection, dict[str, Any]]) -> None:
    with pytest.raises(LookupError):
        get_incident_packet("20990101X99999", loaded[0])


def test_packets_module_never_touches_gold() -> None:
    source = Path(packets.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    assert not any("gold" in name for name in imported), imported

    lowered = source.lower()
    for forbidden in ("narr_accf", "narr_cause", "narr_inc", "findings", "labels", "cm_inpc"):
        assert forbidden not in lowered, forbidden
    queries = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and "SELECT" in n.value
    ]
    assert queries
    for query in queries:
        assert set(re.findall(r"narr_\w+", query)) <= {"narr_accp_safe"}, query


def test_packet_narratives_have_headings_split(
    loaded: tuple[Connection, dict[str, Any]], fixture_data: FixtureData
) -> None:
    """Packets use the same heading split as passages, so both tokenize alike."""
    conn, _ = loaded
    glued = re.compile(r"(HISTORY OF FLIGHT|INFORMATION)[A-Za-z0-9]")
    raw_glued = 0
    for ev_id in fixture_data.ev_ids:
        for a in get_incident_packet(ev_id, conn).aircraft:
            assert not (a.narr_accp and glued.search(a.narr_accp)), ev_id
    for n in fixture_data.narratives:
        raw_glued += bool(glued.search(n["narr_accp"]))
    assert raw_glued > 0  # the raw export does glue headings, so the split matters
