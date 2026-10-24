"""Block identity and text stability across scan passes (no pixel state)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

from .geometry import Box, Rect, iou

STABILITY_WINDOWS_S = {"safe": 0.300, "fast": 0.150}


class BlockState(Enum):
    PENDING = "pending"
    STABLE = "stable"


@dataclass
class TrackBlock:
    id: int
    rect: Rect
    text: str = ""
    state: BlockState = BlockState.PENDING
    first_seen: float = 0.0
    last_seen: float = 0.0
    line_px: int = 0


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def match_score(a: Rect, b: Rect) -> float:
    """IoU, with a floor for containment (a paragraph that grows a line)."""
    score = iou(a, b)
    if score >= 0.5:
        return score
    inter_w = min(a.right, b.right) - max(a.x, b.x)
    inter_h = min(a.bottom, b.bottom) - max(a.y, b.y)
    if inter_w <= 0 or inter_h <= 0:
        return score
    covers = inter_w / min(a.w, b.w) >= 0.9 and inter_h / min(a.h, b.h) >= 0.9
    return 0.6 if covers else score


class BlockMemory:
    """Exact-text block memory: the only validity rule is text equality."""

    def __init__(self, profile: str = "safe") -> None:
        self.stable_after_s = STABILITY_WINDOWS_S.get(
            profile, STABILITY_WINDOWS_S["safe"]
        )
        self._blocks: list[TrackBlock] = []
        self._next_id = 1
        self._lock = threading.RLock()

    def set_profile(self, profile: str) -> None:
        with self._lock:
            self.stable_after_s = STABILITY_WINDOWS_S.get(
                profile, STABILITY_WINDOWS_S["safe"]
            )

    def reset(self) -> None:
        with self._lock:
            self._blocks.clear()
            self._next_id = 1

    def observe(self, boxes: list[Box], now: float, ttl_s: float) -> list[TrackBlock]:
        with self._lock:
            matched: set[int] = set()
            for box in boxes:
                best: TrackBlock | None = None
                best_score = 0.5
                for block in self._blocks:
                    if block.id in matched:
                        continue
                    score = match_score(block.rect, box.rect)
                    if score > best_score:
                        best, best_score = block, score
                if best is None:
                    best = TrackBlock(
                        id=self._next_id, rect=box.rect, first_seen=now, last_seen=now
                    )
                    self._next_id += 1
                    self._blocks.append(best)
                matched.add(best.id)
                best.rect = box.rect
                best.last_seen = now
                if box.line_px:
                    best.line_px = box.line_px
                self._apply_text(best, box.text, now)

            kept: list[TrackBlock] = []
            matched_rects = [b.rect for b in self._blocks if b.id in matched]
            for block in self._blocks:
                if block.id in matched:
                    kept.append(block)
                    continue
                if now - block.last_seen > ttl_s:
                    continue
                # a moved/re-read paragraph must not ghost at the old rect
                if any(iou(block.rect, rect) >= 0.1 for rect in matched_rects):
                    continue
                kept.append(block)
            self._blocks = kept
            return list(kept)

    def _apply_text(self, block: TrackBlock, text: str, now: float) -> None:
        new = _normalize(text)
        old = _normalize(block.text)
        if old and new == old:
            if now - block.first_seen >= self.stable_after_s:
                block.state = BlockState.STABLE
            return
        block.text = text
        block.first_seen = now
        block.state = BlockState.PENDING
