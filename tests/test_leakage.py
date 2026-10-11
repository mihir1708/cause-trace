"""Unit tests for the narrative leakage gate."""

from causetrace.leakage import sanitize_factual_narrative

FACT_A = "The pilot departed from runway 13 at 0830 with full fuel tanks and two passengers aboard."
FACT_B = "Witnesses observed the airplane climb to about 300 feet before the engine lost power."
FACT_C = "The airplane came to rest inverted in a field about half a mile south of the airport."


def test_identical_text_is_withheld() -> None:
    accp = f"{FACT_A} {FACT_B}"
    gate = sanitize_factual_narrative(accp, [accp, "Fuel exhaustion."])
    assert gate.text is None
    assert gate.sentences_total == gate.sentences_removed == 2


def test_disjoint_text_is_kept_with_paragraphs() -> None:
    accp = f"{FACT_A} {FACT_B}\n\n{FACT_C}"
    gate = sanitize_factual_narrative(accp, ["The carburetor icing was not detected in time."])
    assert gate.text == f"{FACT_A} {FACT_B}\n\n{FACT_C}"
    assert gate.sentences_removed == 0


def test_copied_sentence_is_removed() -> None:
    accp = f"{FACT_A} {FACT_B} {FACT_C}"
    analysis = (
        "The analysis notes that witnesses observed the airplane climb to about 300 feet"
        " before the engine lost power, likely due to icing."
    )
    gate = sanitize_factual_narrative(accp, [analysis])
    assert gate.text == f"{FACT_A} {FACT_C}"
    assert gate.sentences_removed == 1


def test_short_gold_phrase_is_matched_whole() -> None:
    accp = f"{FACT_A} Examination revealed fuel exhaustion in both tanks. {FACT_C}"
    gate = sanitize_factual_narrative(accp, ["Fuel exhaustion."])
    assert gate.text == f"{FACT_A} {FACT_C}"


def test_run_spanning_two_sentences_removes_both() -> None:
    accp = f"{FACT_A} {FACT_B} {FACT_C}"
    gold = "about 300 feet before the engine lost power. The airplane came to rest inverted"
    gate = sanitize_factual_narrative(accp, [gold])
    assert gate.text == FACT_A


def test_empty_inputs() -> None:
    assert sanitize_factual_narrative(None, []).text is None
    assert sanitize_factual_narrative(FACT_A, [None, ""]).text == FACT_A


def test_short_copied_gold_sentence_is_removed() -> None:
    accp = f"{FACT_A} The density altitude was calculated at 7,965 feet. {FACT_C}"
    analysis = (
        "The pilot did not account for the high density altitude when planning the takeoff"
        " from the mountain airstrip. The density altitude was calculated at 7,965 feet."
    )
    gate = sanitize_factual_narrative(accp, [analysis])
    assert gate.text == f"{FACT_A} {FACT_C}"


def test_very_short_gold_sentence_is_not_blocked_alone() -> None:
    analysis = (
        "The pilot did not account for the high density altitude when planning the takeoff"
        " from the mountain airstrip. He survived."
    )
    accp = f"{FACT_A} He survived. {FACT_C}"
    assert sanitize_factual_narrative(accp, [analysis]).text == accp
