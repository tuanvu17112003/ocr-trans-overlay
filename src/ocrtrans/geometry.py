"""Integer geometry in physical pixels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def contains_point(self, px: int, py: int) -> bool:
        return self.x <= px < self.right and self.y <= py < self.bottom

    def intersects(self, other: "Rect") -> bool:
        return intersects(self, other)


@dataclass(frozen=True)
class Box:
    rect: Rect
    text: str
    confidence: float
    line_px: int = 0


def intersects(a: Rect, b: Rect) -> bool:
    return a.x < b.right and b.x < a.right and a.y < b.bottom and b.y < a.bottom


def iou(a: Rect, b: Rect) -> float:
    if not intersects(a, b):
        return 0.0
    inter_w = min(a.right, b.right) - max(a.x, b.x)
    inter_h = min(a.bottom, b.bottom) - max(a.y, b.y)
    inter = inter_w * inter_h
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def union_all(rects: list[Rect]) -> Rect:
    if not rects:
        return Rect(0, 0, 0, 0)
    left = min(r.x for r in rects)
    top = min(r.y for r in rects)
    right = max(r.right for r in rects)
    bottom = max(r.bottom for r in rects)
    return Rect(left, top, right - left, bottom - top)


def clamp(rect: Rect, bounds: Rect) -> Rect:
    x = max(bounds.x, min(rect.x, bounds.right))
    y = max(bounds.y, min(rect.y, bounds.bottom))
    right = max(x, min(rect.right, bounds.right))
    bottom = max(y, min(rect.bottom, bounds.bottom))
    return Rect(x, y, right - x, bottom - y)


def expand(rect: Rect, px: int) -> Rect:
    return Rect(rect.x - px, rect.y - px, rect.w + 2 * px, rect.h + 2 * px)
