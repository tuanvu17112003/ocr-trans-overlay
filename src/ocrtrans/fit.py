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
    """Break at word boundaries with the least ragged layout.

    Dynamic programming over the words; the last line is free except that a
    single-word last line is penalised so no orphan is left behind. Words
    wider than the box are hard-broken by character.
    """
    words = text.split()
    if not words:
        return [""]

    atoms: list[str] = []
    spaced: list[bool] = []  # space before this atom?
    for word in words:
        if measure.measure(word, font)[0] <= width:
            atoms.append(word)
            spaced.append(True)
            continue
        chunk = ""
        first = True
        for char in word:
            if chunk and measure.measure(chunk + char, font)[0] > width:
                atoms.append(chunk)
                spaced.append(first)
                first = False
                chunk = char
            else:
                chunk += char
        if chunk:
            atoms.append(chunk)
            spaced.append(first)

    widths = [measure.measure(atom, font)[0] for atom in atoms]
    space = measure.measure(" ", font)[0]
    count = len(atoms)
    orphan_penalty = float(width * width * (count + 1))
    best = [0.0] + [float("inf")] * count
    start = [0] * (count + 1)
    for end in range(1, count + 1):
        line_w = widths[end - 1]
        for begin in range(end - 1, -1, -1):
            if begin < end - 1:
                line_w += widths[begin] + (space if spaced[begin + 1] else 0)
            if line_w > width:
                break
            if end == count:
                cost = orphan_penalty if end - begin == 1 and count > 1 else 0.0
            else:
                cost = float((width - line_w) ** 2)
            total = best[begin] + cost
            if total < best[end]:
                best[end] = total
                start[end] = begin
    if best[count] == float("inf"):
        return atoms  # one character is wider than the box

    lines: list[str] = []
    end = count
    while end > 0:
        begin = start[end]
        lines.append(
            "".join(
                (" " if spaced[index] and index > begin else "") + atoms[index]
                for index in range(begin, end)
            )
        )
        end = begin
    return lines[::-1]


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
    limit_bottom: int | None = None,
) -> FitResult:
    """Size from the source line height, shrink inside the box, then ellipsize.

    The painted box never leaves the source rectangle horizontally. limit_bottom
    lets the cover grow downwards (a longer translation) up to that screen row.
    line_px is one OCR line height; 0 falls back to the block height.
    """
    text = " ".join(text.split())
    anchor = line_px if line_px > 0 else source.h * 0.8
    base_font = max(min_font_px, min(round(anchor * 0.8), source.h))
    width = max(1, min(source.w, screen.right - source.x))
    bottom = source.bottom if limit_bottom is None else max(source.bottom, limit_bottom)
    max_h = max(1, bottom - source.y)

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


