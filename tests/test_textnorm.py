"""Shared normalisation: the three keys must stay distinct and safe."""

from ocrtrans.textnorm import (
    change_class,
    compare_key,
    digits_of,
    has_digit_change,
    normalize,
    scan_fingerprint,
    similarity,
)


def test_normalize_is_nfc_and_collapses_space():
    # Vietnamese can arrive decomposed (NFD) from a model; NFC must recombine it.
    decomposed = "Tie\u0302\u0301ng Vie\u0302\u0323t"
    assert normalize(decomposed) == "Tiếng Việt"
    assert normalize("  a\t\tb \n c ") == "a b c"
    assert normalize("") == ""


def test_normalize_drops_control_characters():
    assert normalize("a\x00b\x07c") == "abc"


def test_compare_key_folds_case_only():
    assert compare_key("Open  The GATE") == "open the gate"
    # compare_key must NOT fold confusables: that is the scheduling key's job.
    assert compare_key("100 HP") != compare_key("l00 HP")


def test_scan_fingerprint_folds_confusables_and_trailing_punctuation():
    assert scan_fingerprint("Open the gate, traveler.") == scan_fingerprint(
        "Open the gate, traveler"
    )
    assert scan_fingerprint("l00 HP") == scan_fingerprint("100 HP")
    assert scan_fingerprint("rnove") == scan_fingerprint("move")


def test_scan_fingerprint_is_not_a_validity_key():
    # The whole point: the fingerprint says "same scheduling bucket" while the
    # validity key says "different text". Keeping them different is the safety
    # property that stops 100 HP / 10 HP from sharing a translation.
    assert compare_key("You have 100 HP") != compare_key("You have 10 HP")
    # A wobble that the fingerprint folds away still counts as "same".
    assert scan_fingerprint("l00 HP") == scan_fingerprint("100 HP")


def test_digits_of_and_has_digit_change():
    assert digits_of("You have 100 HP at 12:45") == "1001245"
    assert has_digit_change("100 HP", "10 HP") is True
    assert has_digit_change("Open the gate", "Open the gates") is False


def test_similarity_bounds():
    assert similarity("hello world", "hello world") == 1.0
    assert similarity("", "") == 1.0
    assert similarity("abc", "") == 0.0
    assert 0.8 < similarity("Open the gate, traveler.", "Open the gate traveler") < 1.0


def test_change_class_same():
    # Spacing, case and a trailing period are OCR wobble, not an edit.
    assert change_class("Open the gate, traveler.", "  open   the gate, traveler ") == "same"
    assert change_class("HP 100", "HP 100.") == "same"


def test_change_class_digit_change_is_always_semantic():
    assert change_class("You have 100 HP", "You have 10 HP") == "semantic"
    assert change_class("Wave 3", "Wave 4") == "semantic"


def test_change_class_cosmetic_wobble():
    # A dropped comma and a doubled letter inside a long sentence: the line
    # means the same thing, so the cover may stay while it is retranslated.
    assert (
        change_class(
            "Open the gate, traveler.",
            "Open the gate traveller",
        )
        == "cosmetic"
    )


def test_change_class_short_lines_are_never_cosmetic():
    assert change_class("Yes", "No") == "semantic"
    assert change_class("HP 100", "HP 101") == "semantic"


def test_change_class_different_sentence_is_semantic():
    assert change_class("Open the gate, traveler.", "The gate is locked.") == "semantic"

def test_change_class_word_swap_is_semantic_not_cosmetic():
    # Same sentence shape, one real word changed: this must never keep the
    # previous translation on screen.
    assert change_class("The gate is locked.", "The gate is open.") == "semantic"
    assert change_class("Press F to interact", "Press E to interact") == "semantic"


def test_change_class_long_sentence_typo_is_cosmetic():
    assert (
        change_class(
            "Oh, what am I supposed to do now?",
            "Oh, what am I supposed to do now ?",
        )
        == "same"
    )
