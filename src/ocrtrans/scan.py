"""Scan pacing: when the next OCR pass may start, and the block TTL."""

from __future__ import annotations

import cv2
import numpy as np

MIN_INTERVAL_S = 0.25
OCR_HEADROOM = 1.5
EMPTY_BACKOFF_S = 1.0
EMPTY_PASSES = 3
TTL_MIN_S = 0.4
TTL_MAX_S = 1.0
WAKE_SAMPLE_S = 0.25
STEADY_INTERVAL_S = 0.5


class ScanBudget:
    def __init__(self, fps: int = 15) -> None:
        self.fps = fps

    def set_fps(self, fps: int) -> None:
        self.fps = max(1, fps)

    def interval(self, last_ocr_ms: float, empty_count: int = 0, wake: bool = False) -> float:
        base = max(1.0 / self.fps, MIN_INTERVAL_S, OCR_HEADROOM * last_ocr_ms / 1000.0)
        if empty_count >= EMPTY_PASSES and not wake:
            base = max(base, EMPTY_BACKOFF_S)
        return base

    def ttl(self, interval_s: float) -> float:
        return min(max(2.0 * interval_s, TTL_MIN_S), TTL_MAX_S)


def frame_changed(
    prev: np.ndarray | None,
    frame: np.ndarray,
    diff_luma: int = 12,
    threshold: float = 0.01,
) -> tuple[bool, np.ndarray]:
    """Cheap 320 px luma diff: a wake signal only, never a scheduler."""
    gray = frame
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w > 320:
        gray = cv2.resize(
            gray, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA
        )
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if prev is None or prev.shape != gray.shape:
        return False, gray
    diff = cv2.absdiff(gray, prev)
    return float((diff > diff_luma).mean()) > threshold, gray
