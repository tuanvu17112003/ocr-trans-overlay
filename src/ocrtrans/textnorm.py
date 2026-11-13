"""Text normalisation shared by scheduling, identity and display.

Three different questions are asked of the same OCR string, and mixing them up
is what made the old build strobe. They are separated here on purpose:

```
normalize()        -> NFC, no control chars, single spaces.  Cache keys, logging.
compare_key()      -> + casefold.                             "is this the same text?"
scan_fingerprint() -> + confusable folding, trailing punctuation dropped.
                                                              SCHEDULING ONLY.
change_class()     -> same | cosmetic | semantic.             Display hysteresis.
```

The one rule that must never bend: a difference in **digits** is always
semantic. "You have 100 HP" and "You have 10 HP" are ~95% similar and mean
different things, so no hysteresis may ever keep the old translation for them.
`scan_fingerprint` exists purely to decide how often to OCR; it is never
consulted when deciding whether a translation is still valid.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

__all__ = [
    "ChangeClass",
    "change_class",
    "compare_key",
    "digits_of",
    "has_digit_change",
    "normalize",
    "scan_fingerprint",
    "similarity",
]

ChangeClass = Literal["same", "cosmetic", "semantic"]

# Characters an OCR engine swaps on a low-contrast or moving background. Folding
# them is safe for *scheduling* because a real edit between these classes is
# rare, and it is unsafe for validity, which is why only the fingerprint uses it.
_CONFUSABLE = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "i": "l",
        "|": "l",
        "5": "s",
        "2": "z",
        "8": "b",
    }
)
_CONFUSABLE_SEQ = [(re.compile(r"rn"), "m"), (re.compile(r"vv"), "w"), (re.compile(r"cl"), "d")]

_TRAILING_PUNCT = ".!?,;:…\"'”’)"

_COSMETIC_MIN_SIMILARITY = 0.85
_COSMETIC_MIN_WORDS = 3
_COSMETIC_MIN_TOKEN_AGREEMENT = 0.6
_COSMETIC_MIN_CHANGED_TOKEN_SIMILARITY = 0.6


def normalize(text: str) -> str:
    """NFC, control characters removed, whitespace collapsed and trimmed."""
    if not text:
        return ""
    # NFC first: Vietnamese from a model can arrive decomposed, and decomposed
    # glyphs measure differently in QFontMetrics, which breaks wrapping.
    cleaned = unicodedata.normalize("NFC", text)
    # Control characters include \t and \n, which are word separators an OCR
    # engine emits for a gap. Converting them (rather than dropping them) keeps
    # "a\tb" as two words instead of gluing them into "ab".
    cleaned = "".join(
        " " if ch in "\t\n\r\f\v" else ch
        for ch in cleaned
        if ch in "\t\n\r\f\v" or unicodedata.category(ch)[0] != "C"
    )
    return " ".join(cleaned.split())


def compare_key(text: str) -> str:
    """Identity key: normalised, casefolded. Two reads with the same key are the
    same text for the purpose of the overlay."""
    return normalize(text).casefold()


def scan_fingerprint(text: str) -> str:
    """Scheduling-only key: folds confusable glyphs and drops trailing noise.

    Never use this to decide that a translation is still valid; it is only used
    to decide whether the scan cadence may slow down.
    """
    key = compare_key(text)
    if not key:
        return ""
    key = key.translate(_CONFUSABLE)
    for pattern, replacement in _CONFUSABLE_SEQ:
        key = pattern.sub(replacement, key)
    key = key.rstrip(_TRAILING_PUNCT + " ")
    return " ".join(key.split())


def digits_of(text: str) -> str:
    """Every digit in order, so "100" vs "10" is a difference, not a similarity."""
    return "".join(ch for ch in normalize(text) if ch.isdigit())


def has_digit_change(a: str, b: str) -> bool:
    return digits_of(a) != digits_of(b)


def similarity(a: str, b: str) -> float:
    """Cheap character-level ratio for display hysteresis only (no difflib)."""
    x, y = compare_key(a), compare_key(b)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0
    # Levenshtein distance, two-row rolling buffer. Strings here are at most a
    # few hundred characters, so O(n*m) at ~100 chars is ~10k cell updates.
    if len(x) < len(y):
        x, y = y, x
    previous = list(range(len(y) + 1))
    for i, cx in enumerate(x, start=1):
        current = [i]
        for j, cy in enumerate(y, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if cx == cy else 1),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(x), len(y))


_TOKEN_STRIP = ".!?,;:…\"'”’)("


def _tokens(text: str) -> list[str]:
    return [word.strip(_TOKEN_STRIP) for word in compare_key(text).split()]


def _word_agreement(a: str, b: str) -> float:
    """Share of positions that hold the same word, matched up to a one-word shift."""
    left, right = _tokens(a), _tokens(b)
    if not left or not right or abs(len(left) - len(right)) > 1:
        return 0.0
    width = min(len(left), len(right))
    same = sum(1 for i in range(width) if left[i] == right[i])
    return same / max(len(left), len(right))


def _changed_tokens_are_typos(a: str, b: str) -> bool:
    """Every word that differs must be a near-spelling of its counterpart.

    This is the guard that separates a wobble ("traveler" -> "traveller") from a
    real edit ("locked" -> "open") on lines of the same shape.
    """
    left, right = _tokens(a), _tokens(b)
    if abs(len(left) - len(right)) > 1:
        return False
    for first, second in zip(left, right):
        if first == second:
            continue
        if similarity(first, second) < _COSMETIC_MIN_CHANGED_TOKEN_SIMILARITY:
            return False
    return True


def change_class(old: str, new: str) -> ChangeClass:
    """Classify an OCR-to-OCR transition for the overlay.

    same      the same text after normalisation: do nothing at all.
    cosmetic  a short wobble in a long line: keep the cover, retranslate quietly.
    semantic  a real change (any digit change always lands here): cut at once.
    """
    # Any digit difference is semantic, before any folding is considered. This
    # is the invariant the whole design rests on: "100 HP" -> "10 HP" must never
    # keep the translation of "100 HP".
    if has_digit_change(old, new):
        return "semantic"
    # Nothing to do when only the OCR wobble differs (trailing punctuation,
    # confusable glyphs, spacing). Judged on the scheduling key, never on the
    # validity key, because the validity key is deliberately strict.
    if scan_fingerprint(old) == scan_fingerprint(new):
        return "same"
    old_words = _tokens(old)
    new_words = _tokens(new)
    # A short line has no redundancy to lean on: "Yes" -> "No" is an edit.
    if min(len(old_words), len(new_words)) < _COSMETIC_MIN_WORDS:
        return "semantic"
    # A whole new sentence is semantic regardless of the character ratio.
    if similarity(old, new) < _COSMETIC_MIN_SIMILARITY:
        return "semantic"
    if _word_agreement(old, new) < _COSMETIC_MIN_TOKEN_AGREEMENT:
        return "semantic"
    # Every differing word must be a near-spelling of its counterpart, so
    # "the gate is locked" -> "the gate is open" stays semantic even though the
    # line shape and the character ratio barely move.
    if not _changed_tokens_are_typos(old, new):
        return "semantic"
    return "cosmetic"