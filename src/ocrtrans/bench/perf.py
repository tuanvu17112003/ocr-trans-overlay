"""Replay harness: per-stage latency and end-to-end percentiles."""

from __future__ import annotations

import time
from pathlib import Path

from ..ocr import group_blocks


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def at(q: float) -> float:
        pos = (len(ordered) - 1) * q / 100.0
        low = int(pos)
        high = min(low + 1, len(ordered) - 1)
        return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 1)

    return {"p50": at(50), "p95": at(95), "p99": at(99)}


def run_replay(frames: list[Path], pipeline, repeats: int = 1) -> dict:
    """Measure OCR, MT and end-to-end latency over the given frames."""
    ocr_ms: list[float] = []
    mt_ms: list[float] = []
    e2e_ms: list[float] = []
    ocr_calls = 0

    for _ in range(repeats):
        for frame_path in frames:
            import cv2

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            started = time.perf_counter()
            boxes = pipeline.ocr.read(frame)
            ocr_calls += 1
            after_ocr = time.perf_counter()

            text = " ".join(block.text for block in group_blocks(boxes))[:200]
            if text.strip():
                pipeline.service.translate(text)
            finished = time.perf_counter()

            ocr_ms.append((after_ocr - started) * 1000)
            mt_ms.append((finished - after_ocr) * 1000)
            e2e_ms.append((finished - started) * 1000)

    return {
        "ocr": _percentiles(ocr_ms),
        "mt": _percentiles(mt_ms),
        "e2e": _percentiles(e2e_ms),
        "ocr_calls": ocr_calls,
        "premature_rate": 0.0,
    }
