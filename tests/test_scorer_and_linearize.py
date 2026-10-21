"""Fast tests (no model download): aggregation logic and JSON linearization."""

import pytest

from src.data.linearize import business_facts, format_time, hours_sentence, parse_music
from src.verify.nli import NLIResult
from src.verify.scorer import aggregate


def r(e, n, c, span=(0, 10)):
    return NLIResult(e, n, c, span)


# --- aggregation -------------------------------------------------------------------------------


def test_any_passage_entailing_means_supported():
    v = aggregate([[r(0.1, 0.8, 0.1)], [r(0.9, 0.05, 0.05)]])
    assert v.label == "supported" and v.entail_at[0] == 1 and not v.conflicting


def test_contradiction_without_entailment():
    assert aggregate([[r(0.1, 0.2, 0.7)], [r(0.2, 0.7, 0.1)]]).label == "contradicted"


def test_nothing_above_threshold_is_unsupported():
    assert aggregate([[r(0.3, 0.4, 0.3)]]).label == "unsupported"


@pytest.mark.parametrize(
    "tie_break, expected",
    [("max_prob", "contradicted"), ("supported", "supported"), ("contradicted", "contradicted")],
)
def test_conflicting_evidence_uses_tie_break(tie_break, expected):
    # ex. 9 shape: one window entails, a later one contradicts more strongly
    v = aggregate([[r(0.7, 0.2, 0.1, (0, 100)), r(0.05, 0.05, 0.9, (900, 1000))]], tie_break=tie_break)
    assert v.conflicting and v.label == expected
    assert v.contradict_at == (0, (900, 1000))


# --- linearization -----------------------------------------------------------------------------


def test_format_time():
    assert format_time("20:30") == "8:30 PM"
    assert format_time("0:0") == "12:00 AM"
    assert format_time("12:0") == "12:00 PM"
    assert format_time("8:0") == "8:00 AM"


def test_zero_hours_rules():
    assert hours_sentence("X", "Monday", "0:0-0:0") == "X is closed on Monday."
    assert "0:00 to 0:00" in hours_sentence("X", "Monday", "0:0-0:0", zero_rule="literal")


def test_music_string_with_stripped_u_is_repaired():
    parsed = parse_music("{dj: None, live: Tre, jkebox: False, backgrond_msic: False}")
    assert parsed == {"dj": None, "live": True, "jukebox": False, "background_music": False}


def test_nulls_are_omitted_and_false_is_negated():
    info = {
        "name": "Cal Taco",
        "address": "7320 Hollister Ave",
        "city": "Goleta",
        "state": "CA",
        "categories": "Burgers",
        "business_stars": 4.0,
        "hours": {"Monday": "0:0-0:0", "Tuesday": "8:0-20:30"},
        "attributes": {
            "BusinessParking": {"street": False, "lot": True, "garage": None},
            "WiFi": "no",
            "Music": None,
            "Ambience": {"casual": True, "trendy": False, "classy": None},
        },
    }
    text = " ".join(business_facts(info))
    assert "Cal Taco is closed on Monday." in text
    assert "On Tuesday, Cal Taco is open from 8:00 AM to 8:30 PM." in text
    assert "does not have street parking" in text and "has a parking lot" in text
    assert "The ambience at Cal Taco is not trendy." in text
    # nulls never appear, in either polarity
    for absent in ("garage", "classy", "music", "Wednesday"):
        assert absent not in text


# --- claim-level evaluation ----------------------------------------------------------------------


def test_sentence_is_flagged_by_its_worst_claim_and_empty_sentences_never():
    from src.data.ragtruth import HallucinationSpan, RagTruthExample
    from src.eval.metrics import evaluate

    resp = "Good part. Bad part here."
    e = RagTruthExample("x", "s", "QA", "MARCO", "m", 0.7, "test", "good", resp, "", None, "",
                        spans=[HallucinationSpan(11, 25, "Bad part here.", "Evident Conflict", False, False, None)])
    rec = {"id": "x", "sentences": [
        {"start": 0, "end": 10, "claims": []},  # e.g. "Sure!" -> no claims -> not flagged
        {"start": 11, "end": 25, "claims": [{"per_passage": [[0.9, 0.0]]}, {"per_passage": [[0.0, 0.9]]}]},
    ]}
    m = evaluate([rec], {"x": e}, t_entail=0.5)
    assert m["sentence_level"]["all"]["tp"] == 1 and m["sentence_level"]["all"]["fp"] == 0
    assert m["type_confusion"] == {"conflict -> contradicted": 1}
