"""Translation quality metrics (chrF++ and optional COMET)."""

from __future__ import annotations

import re
from typing import Sequence

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")


def score_quality(
    pairs: Sequence[tuple[str, str]],
    refs: Sequence[str] | None = None,
) -> dict:
    """pairs are (source, hypothesis); refs enables chrF++."""
    total = len(pairs)
    if total == 0:
        return {
            "chrf_pp": None,
            "comet": None,
            "number_preserved": 0.0,
            "proper_noun_preserved": 0.0,
            "length_ratio": 0.0,
            "empty_count": 0,
        }

    numbers = sum(
        1 for source, hyp in pairs if set(NUMBER.findall(source)) <= set(NUMBER.findall(hyp))
    )
    proper = sum(1 for source, hyp in pairs if _proper_preserved(source, hyp))
    length = sum(len(hyp) / max(1, len(source)) for source, hyp in pairs) / total
    empty = sum(1 for _, hyp in pairs if not hyp.strip())

    chrf = None
    if refs is not None:
        try:
            from sacrebleu.metrics import CHRF

            chrf = round(
                CHRF(word_order=2).corpus_score([hyp for _, hyp in pairs], [list(refs)]).score, 2
            )
        except Exception:  # noqa: BLE001 - sacrebleu is optional
            chrf = None

    return {
        "chrf_pp": chrf,
        "comet": None,  # unbabel-comet is optional and slow; skipped by default
        "number_preserved": numbers / total,
        "proper_noun_preserved": proper / total,
        "length_ratio": round(length, 3),
        "empty_count": empty,
    }


def _proper_preserved(source: str, hyp: str) -> bool:
    tokens = source.split()
    names = PROPER.findall(" ".join(tokens[1:]))
    return all(name in hyp for name in names)
