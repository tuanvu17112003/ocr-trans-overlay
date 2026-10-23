"""Fit translated text into (or around) the source rectangle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .geometry import Rect, intersects


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
    base_ratio: float = 0.8,
    max_growth: float = 1.8,
    min_font_px: int = 12,
    obstacles: Sequence[Rect] = (),
) -> FitResult:
    """Wrap at source width, grow down, widen, shrink to floor, then ellipsize."""
    text = " ".join(text.split())
    base_font = max(min_font_px, round(source.h * base_ratio))
    max_h = max(source.h, min(int(source.h * max_growth), screen.bottom - source.y))
    widths = [source.w, min(int(source.w * 1.3), screen.right - source.x)]
    widths = sorted({max(1, w) for w in widths})

    def build(font: int, width: int, height: int) -> Rect:
        rect = Rect(source.x, source.y, width, max(source.h, height))
        if rect.bottom > screen.bottom:
            rect = Rect(rect.x, screen.bottom - rect.h, rect.w, rect.h)
        if source.y >= screen.y + screen.h / 2:  # subtitle: grow upward
            rect = Rect(rect.x, max(screen.y, source.bottom - rect.h), rect.w, rect.h)
        return rect

    def blocked(rect: Rect) -> bool:
        return any(intersects(rect, obstacle) for obstacle in obstacles)

    for width in widths:
        lines = _wrap(text, width, base_font, measure)
        height = len(lines) * measure.line_height(base_font)
        rect = build(base_font, width, height)
        if height <= max_h and not blocked(rect):
            return FitResult(rect, base_font, lines, False)

    for font in range(base_font - 1, min_font_px - 1, -1):
        for width in widths:
            lines = _wrap(text, width, font, measure)
            height = len(lines) * measure.line_height(font)
            rect = build(font, width, height)
            if height <= max_h and not blocked(rect):
                return FitResult(rect, font, lines, False)

    font = min_font_px
    width = widths[-1]
    line_h = measure.line_height(font)
    lines = _wrap(text, width, font, measure)
    max_lines = max(1, max_h // line_h)
    truncated = len(lines) > max_lines
    lines = lines[:max_lines]
    if lines:
        lines[-1] = _ellipsize(lines[-1], width, font, measure)
    height = len(lines) * line_h
    return FitResult(build(font, width, height), font, lines, truncated)
