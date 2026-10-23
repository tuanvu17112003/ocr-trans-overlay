"""Fit translated text inside the source rectangle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .geometry import Rect


class Measure(Protocol):
    def measure(self, text: str, font_px: int) -> tuple[int, int]: ...

    def line_height(self, font_px: int) -> int: ...


@dataclass
class FitResult:
    rect: Rect
    font_px: int
    lines: list[str]
    truncated: bool


def _wrap(text: str, width: int, font: int, measure: Measure) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if measure.measure(candidate, font)[0] <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        if measure.measure(word, font)[0] <= width:
            current = word
            continue
        # one word wider than the line: hard-break it
        chunk = ""
        for char in word:
            if not chunk or measure.measure(chunk + char, font)[0] <= width:
                chunk += char
            else:
                lines.append(chunk)
                chunk = char
        current = chunk
    if current:
        lines.append(current)
    return lines


def _ellipsize(line: str, width: int, font: int, measure: Measure) -> str:
    if measure.measure(line, font)[0] <= width:
        return line
    trimmed = line
    while trimmed and measure.measure(trimmed + "…", font)[0] > width:
        trimmed = trimmed[:-1]
    return trimmed.rstrip() + "…"


def fit_text(
    text: str,
    source: Rect,
    screen: Rect,
    measure: Measure,
    *,
    line_px: int = 0,
    min_font_px: int = 12,
) -> FitResult:
    """Size from the source line height, shrink inside the box, then ellipsize.

    The painted box never leaves the source rectangle: no widening, no vertical
    growth. line_px is one OCR line height; 0 falls back to the block height.
    """
    text = " ".join(text.split())
    anchor = line_px if line_px > 0 else source.h * 0.8
    base_font = max(min_font_px, min(round(anchor * 0.8), source.h))
    width = max(1, min(source.w, screen.right - source.x))
    max_h = max(1, source.h)

    def rect_for(height: int) -> Rect:
        return Rect(source.x, source.y, width, max(1, min(max_h, height)))

    for font in range(base_font, min_font_px - 1, -1):
        lines = _wrap(text, width, font, measure)
        height = len(lines) * measure.line_height(font)
        rect = rect_for(height)
        if height <= max_h:
            return FitResult(rect, font, lines, False)

    font = min_font_px
    line_h = measure.line_height(font)
    lines = _wrap(text, width, font, measure)
    max_lines = max(1, max_h // line_h)
    truncated = len(lines) > max_lines
    lines = lines[:max_lines]
    if lines:
        lines[-1] = _ellipsize(lines[-1], width, font, measure)
    height = len(lines) * line_h
    return FitResult(rect_for(height), font, lines, truncated)


