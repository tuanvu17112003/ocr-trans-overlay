"""Candidate gate: reject detector output that is artwork, not text.

The old build let anything through the recogniser and then translated it. On a
synthetic icon grid with no text at all, four blocks reached the translator and
were painted over the game. Raising the detector thresholds did not help: the
same four boxes came through at PaddleOCR defaults (0.6 / 0.3), because the
recogniser really does read "x.O^" out of a texture. The filter has to sit
after detection.

What separates text from artwork is shape, not confidence:

  aspect   a Latin caption line is wide and short. A 60x178 box is impossible
           for one line of text and is exactly what a texture patch produces.
  stroke   stroke width over box height sits in a narrow band for rendered type.
           Wood grain and icon shading fall outside it.
  content  the string has to look like language: enough alphanumerics, at
           least some letters, few box-drawing or symbol runs.

Thresholds were measured on a hand-labelled set of 123 detector boxes
(82 artwork, 41 text) taken from game frames and synthetic scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import cv2
import numpy as np

from .geometry import Box, Rect

Profile = Literal["subtitle", "hud", "all"]

# --- measured thresholds ---------------------------------------------------
MIN_ASPECT = 1.2
MAX_ASPECT = 30.0
MIN_STROKE_OVER_H = 0.02
MAX_STROKE_OVER_H = 0.35
MIN_CHARS = 3
# Measured on the 80-line corpus: min 0.73, p05 0.77, median 0.81. The old 0.6
# let "x.O^ O^III" through at exactly 0.60.
MIN_ALNUM_RATIO = 0.70
MIN_LETTERS_SUBTITLE = 2
# HUD numbers ("10/20", "+37", "12:45") carry no letters; that profile accepts
# two digits instead, at the cost of letting more artwork through.
MIN_DIGITS_HUD = 2

# Rejection reasons, so the counter on the status line is actionable.
DENSITY = "density"
ASPECT = "aspect"
STROKE = "stroke"
CONTENT = "content"
CONFIDENCE = "confidence"
TOO_TALL = "too_tall"
TOO_SMALL = "too_small"

PROFILES = {
    "subtitle": {
        "min_confidence": 0.70,
        "min_density": 3e-4,
        "min_letters": MIN_LETTERS_SUBTITLE,
        "min_digits": 99,
    },
    "hud": {
        "min_confidence": 0.65,
        "min_density": 1e-4,
        "min_letters": 0,
        "min_digits": MIN_DIGITS_HUD,
    },
    "all": {
        "min_confidence": 0.60,
        "min_density": 5e-5,
        "min_letters": 0,
        "min_digits": 0,
    },
}


@dataclass(frozen=True)
class Verdict:
    keep: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.keep


@dataclass
class GateStats:
    kept: int = 0
    rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def count(self, verdict: Verdict) -> None:
        if verdict.keep:
            self.kept += 1
            return
        self.rejected += 1
        self.reasons[verdict.reason] = self.reasons.get(verdict.reason, 0) + 1

    def reset(self) -> None:
        self.kept = 0
        self.rejected = 0
        self.reasons.clear()

    def as_dict(self) -> dict:
        return {"kept": self.kept, "rejected": self.rejected, **self.reasons}


def _letter_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _digit_count(text: str) -> int:
    return sum(1 for ch in text if ch.isdigit())


def text_likeness(text: str, rect: Rect, profile: str = "subtitle") -> Verdict:
    """Judge a recognised box without looking at pixels."""
    settings = PROFILES.get(profile, PROFILES["subtitle"])
    stripped = text.strip()
    if not stripped:
        return Verdict(False, CONTENT)

    alnum = sum(1 for ch in stripped if ch.isalnum())
    if alnum < MIN_CHARS:
        return Verdict(False, CONTENT)
    if alnum / len(stripped) < MIN_ALNUM_RATIO:
        return Verdict(False, CONTENT)

    letters = _letter_count(stripped)
    digits = _digit_count(stripped)
    if letters < settings["min_letters"] and digits < settings["min_digits"]:
        return Verdict(False, CONTENT)

    if rect.h <= 0 or rect.w <= 0:
        return Verdict(False, TOO_SMALL)
    aspect = rect.w / rect.h
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return Verdict(False, ASPECT)

    density = alnum / max(1, rect.w * rect.h)
    if density < settings["min_density"]:
        return Verdict(False, DENSITY)

    return Verdict(True)


def stroke_ratio(image: np.ndarray) -> float:
    """Stroke width divided by box height, from a distance transform.

    DIAGNOSTIC ONLY. Measured on this machine it does *not* separate text from
    artwork, so the gate must not use it:

        rendered text   0.095 - 0.239
        icon grid       0.077 - 0.086
        wood grain      0.100 - 0.149
        white noise     0.046 - 0.064

    The bands overlap completely. It is kept because it is useful when
    diagnosing a specific frame (see bench/stability.md), and because the
    measurement is worth recording so nobody re-adds it expecting a filter.
    """
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height = gray.shape[0]
    if height <= 1:
        return 0.0
    # Ink is the minority class: white type on a dark game HUD and dark type on
    # a light menu both have to work, so the mask is picked by area rather than
    # by assuming one polarity.
    gray = np.ascontiguousarray(gray.astype(np.uint8))
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = bright if bright.mean() <= 127 else 255 - bright
    if ink.mean() < 1.0:
        return 0.0
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    peaks = distance[distance >= 0.6 * distance.max()]
    if peaks.size == 0:
        return 0.0
    return float(2.0 * np.median(peaks) / height)


def crop_for(frame: np.ndarray, rect: Rect) -> np.ndarray:
    """Region-local crop for one box, clipped to the frame."""
    height, width = frame.shape[:2]
    x0 = min(max(0, rect.x), width)
    x1 = min(max(x0, rect.right), width)
    y0 = min(max(0, rect.y), height)
    y1 = min(max(y0, rect.bottom), height)
    return frame[y0:y1, x0:x1]


def filter_candidates(
    boxes: list[Box],
    frame: np.ndarray | None = None,
    *,
    profile: str = "subtitle",
    region: Rect | None = None,
    stats: GateStats | None = None,
) -> list[Box]:
    """Drop boxes that are not text.

    `frame` is accepted for API compatibility and future pixel checks; it is not
    consulted today because the stroke measurement was shown not to discriminate
    (see :func:`stroke_ratio`).
    """
    settings = PROFILES.get(profile, PROFILES["subtitle"])
    # A single line of text taller than half the capture region is a blob, not a
    # caption. The threshold has to stay loose: a 40 px font in a 200 px
    # subtitle strip is 27 % of the region and perfectly legitimate, which is
    # what a 25 % cap got wrong. Aspect already catches the 60x178 artwork box;
    # this only exists for blobs that are wide *and* tall.
    max_height = 0.5 * region.h if region is not None else None
    kept: list[Box] = []
    for box in boxes:
        verdict = text_likeness(box.text, box.rect, profile)
        if verdict.keep and max_height is not None and box.rect.h > max_height:
            verdict = Verdict(False, TOO_TALL)
        if verdict.keep and box.confidence < settings["min_confidence"]:
            verdict = Verdict(False, CONFIDENCE)
        if stats is not None:
            stats.count(verdict)
        if verdict.keep:
            kept.append(box)
    return kept