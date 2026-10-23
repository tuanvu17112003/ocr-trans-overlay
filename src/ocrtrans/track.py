"""Temporal tracker: decides when OCR must run again."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

from .geometry import Box, Rect, expand, iou

STABILITY_WINDOWS = {"safe": 3, "fast": 2}


class BlockState(Enum):
    NEW = auto()
    PENDING = auto()
    STABLE = auto()
    STALE = auto()
    GONE = auto()


@dataclass
class TrackBlock:
    id: int
    rect: Rect
    text: str = ""
    state: BlockState = BlockState.NEW
    stable_passes: int = 0
    miss_count: int = 0
    last_seen_pass: int = 0


@dataclass
class Decision:
    full_detect: bool = False
    crops: list[tuple[int, Rect]] = field(default_factory=list)
    skip: bool = False


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


class TemporalTracker:
    def __init__(
        self,
        profile: str = "safe",
        block_diff_ratio: float = 0.02,
        region_diff_ratio: float = 0.01,
        drift_ratio: float = 0.05,
        scene_cut_ratio: float = 0.70,
        blind_cap_s: float = 1.5,
        refresh_s: float = 1.5,
        detect_throttle_passes: int = 4,
        diff_luma: int = 12,
        samples_per_s: int = 15,
    ) -> None:
        self.window = STABILITY_WINDOWS.get(profile, 3)
        self.block_diff_ratio = block_diff_ratio
        self.region_diff_ratio = region_diff_ratio
        self.drift_ratio = drift_ratio
        self.scene_cut_ratio = scene_cut_ratio
        self.blind_cap_s = blind_cap_s
        self.refresh_s = refresh_s
        self.detect_throttle_passes = max(1, detect_throttle_passes)
        self.diff_luma = diff_luma
        self.samples_per_s = samples_per_s

        self._prev: np.ndarray | None = None
        self._blocks: list[TrackBlock] = []
        self._next_id = 1
        self._pass = 0
        self._passes_since_detect = 0
        self._last_full = 0.0

    def blocks(self) -> list[TrackBlock]:
        return [b for b in self._blocks if b.state is not BlockState.GONE]

    def reset(self) -> None:
        self._blocks.clear()
        self._prev = None
        self._pass = 0
        self._passes_since_detect = 0
        self._last_full = 0.0

    # ---- frame diff -------------------------------------------------------

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        if w > 320:
            frame = cv2.resize(frame, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def update(self, frame: np.ndarray, now: float) -> Decision:
        self._pass += 1
        gray = self._gray(frame)
        if self._prev is None:
            self._prev = gray
            self._passes_since_detect = 0
            self._last_full = now
            return Decision(full_detect=True)

        changed = cv2.absdiff(gray, self._prev) > self.diff_luma
        self._prev = gray
        ratio = float(changed.mean())

        if ratio >= self.scene_cut_ratio:
            self._blocks.clear()
            self._passes_since_detect = 0
            self._last_full = now
            return Decision(full_detect=True)

        self._passes_since_detect += 1
        stable = any(b.state is BlockState.STABLE for b in self._blocks)

        if ratio < self.region_diff_ratio:
            if self._passes_since_detect >= self.detect_throttle_passes and (
                now - self._last_full >= self.blind_cap_s
            ):
                self._passes_since_detect = 0
                self._last_full = now
                return Decision(full_detect=True)
            if not stable and not self._blocks:
                self._passes_since_detect = 0
                self._last_full = now
                return Decision(full_detect=True)
            return Decision(skip=True)

        crops: list[tuple[int, Rect]] = []
        sx = gray.shape[1] / frame.shape[1]
        sy = gray.shape[0] / frame.shape[0]
        for block in self._blocks:
            rect = expand(block.rect, 2)
            rx, ry = int(rect.x * sx), int(rect.y * sy)
            rw, rh = max(1, int(rect.w * sx)), max(1, int(rect.h * sy))
            sub = changed[ry : ry + rh, rx : rx + rw]
            if sub.size and float(sub.mean()) >= self.block_diff_ratio:
                crops.append((block.id, rect))

        if crops:
            return Decision(crops=crops)

        if not stable or ratio >= self.drift_ratio:
            if self._passes_since_detect >= self.detect_throttle_passes:
                self._passes_since_detect = 0
                self._last_full = now
                return Decision(full_detect=True)

        if now - self._last_full >= self.blind_cap_s:
            self._passes_since_detect = 0
            self._last_full = now
            return Decision(full_detect=True)

        return Decision(skip=True)

    # ---- OCR results ------------------------------------------------------

    def observe(
        self,
        boxes: list[Box],
        now: float,
        scope: set[int] | None = None,
    ) -> list[TrackBlock]:
        """scope limits which blocks may be counted missing (crop scans)."""
        matched: set[int] = set()
        for box in boxes:
            best, best_iou = None, 0.5
            for block in self._blocks:
                score = iou(block.rect, box.rect)
                if score >= best_iou:
                    best, best_iou = block, score
            if best is None:
                best = TrackBlock(id=self._next_id, rect=box.rect)
                self._next_id += 1
                self._blocks.append(best)
            matched.add(best.id)
            self._apply_text(best, box.text, now)

        for block in self._blocks:
            if block.id in matched:
                block.miss_count = 0
            elif scope is None or block.id in scope:
                block.miss_count += 1
                if block.miss_count >= 2:
                    block.state = BlockState.GONE

        self._blocks = [b for b in self._blocks if b.state is not BlockState.GONE]
        return self.blocks()

    def _apply_text(self, block: TrackBlock, text: str, now: float) -> None:
        new = _normalize(text)
        old = _normalize(block.text)
        block.last_seen_pass = self._pass
        if not old:
            block.text = text
            block.stable_passes = 1
            block.state = BlockState.PENDING
            return
        if new == old:
            block.stable_passes += 1
        elif new.startswith(old) or old.startswith(new):
            block.text = text
            block.stable_passes = 1
            block.state = BlockState.PENDING
            return
        elif difflib.SequenceMatcher(None, new, old).ratio() >= 0.9:
            block.text = text
            block.stable_passes += 1
        else:
            block.text = text
            block.stable_passes = 1
            block.state = BlockState.PENDING
            return
        if block.stable_passes >= self.window:
            block.state = BlockState.STABLE
        elif block.state is not BlockState.STABLE:
            block.state = BlockState.PENDING
