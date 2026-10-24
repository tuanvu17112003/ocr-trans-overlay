"""OCR engine (RapidOCR/ONNX) and paragraph block grouping."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2

from .geometry import Box, Rect, iou, union_all


@dataclass
class Block:
    rect: Rect
    lines: list[Box] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)

    @property
    def line_px(self) -> int:
        return _median([line.rect.h for line in self.lines]) if self.lines else self.rect.h


def _median(values: list[int]) -> int:
    if not values:
        return 1
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def group_blocks(
    boxes: list[Box],
    max_gap_ratio: float = 0.6,
    min_confidence: float = 0.6,
    max_lines: int = 6,
) -> list[Block]:
    """Merge text lines that sit under each other into paragraph blocks."""
    kept = [b for b in boxes if b.confidence >= min_confidence and b.text.strip()]
    if not kept:
        return []
    kept.sort(key=lambda b: (b.rect.y, b.rect.x))
    line_h = _median([b.rect.h for b in kept])

    groups: list[list[Box]] = []
    for box in kept:
        placed = False
        if groups and len(groups[-1]) < max_lines:
            group = groups[-1]
            last = group[-1]
            v_gap = box.rect.y - last.rect.bottom
            h_gap = box.rect.x - last.rect.right
            same_row = (
                abs(box.rect.y - last.rect.y) <= 0.2 * line_h
                and -0.4 * line_h <= h_gap <= 0.4 * line_h
            )
            same_column = abs(box.rect.x - group[0].rect.x) <= 2 * line_h
            vertical = -0.2 * line_h <= v_gap <= max_gap_ratio * line_h and same_column
            if same_row or vertical:
                group.append(box)
                placed = True
        if not placed:
            groups.append([box])

    blocks = []
    for group in groups:
        group.sort(key=lambda b: (b.rect.y, b.rect.x))
        blocks.append(Block(rect=union_all([b.rect for b in group]), lines=group))
    blocks.sort(key=lambda b: (b.rect.y, b.rect.x))
    return blocks


def _prepare_cuda_dlls() -> None:
    """onnxruntime-gpu finds cuBLAS/cuDNN only inside torch/lib on Windows."""
    try:
        import torch

        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
    except Exception:  # noqa: BLE001 - no torch means CPU-only, reported by load()
        pass


class RapidOcrEngine:
    def __init__(self, use_cuda: bool = True, downscale_width: int = 1280) -> None:
        self.use_cuda = use_cuda
        self.downscale_width = downscale_width
        self._engine = None

    def load(self) -> None:
        _prepare_cuda_dlls()
        from rapidocr import RapidOCR

        try:
            self._engine = RapidOCR(params=self._params(self.use_cuda))
        except Exception:  # noqa: BLE001 - CUDA missing must not kill the app
            self.use_cuda = False
            self._engine = RapidOCR(params=self._params(False))

    @staticmethod
    def _params(use_cuda: bool) -> dict:
        # HUD text is sparse and often faint, so thresholds sit below PaddleOCR's
        # defaults; capping the long side keeps a full-screen region fast enough.
        return {
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
            "Det.limit_type": "max",
            "Det.limit_side_len": 1280,
            "Det.box_thresh": 0.3,
            "Det.thresh": 0.2,
        }

    def _ensure_loaded(self) -> None:
        if self._engine is None:
            self.load()

    def read(self, frame: np.ndarray) -> list[Box]:
        self._ensure_loaded()
        assert self._engine is not None

        scale = 1.0
        img = frame
        if self.downscale_width and frame.shape[1] > self.downscale_width:
            scale = self.downscale_width / frame.shape[1]
            img = cv2.resize(
                frame,
                (self.downscale_width, max(1, round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        out = self._engine(img, use_det=True, use_cls=False, use_rec=True)
        return self._to_boxes(out, scale)

    @staticmethod
    def _to_boxes(out, scale: float) -> list[Box]:
        raw = getattr(out, "boxes", None)
        texts = getattr(out, "txts", None) or ()
        scores = getattr(out, "scores", None) or ()
        if raw is None:
            return []
        boxes: list[Box] = []
        for quad, text, score in zip(raw, texts, scores):
            points = np.asarray(quad).reshape(-1, 2) / scale
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            boxes.append(
                Box(
                    rect=Rect(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0))),
                    text=str(text),
                    confidence=float(score),
                )
            )
        return boxes

    def close(self) -> None:
        self._engine = None


def plan_tiles(
    width: int,
    height: int,
    max_side: int = 1280,
    overlap: int = 64,
    max_tiles: int = 2,
) -> list[Rect]:
    """Region-local tiles for the bounded rescue scan, row-major."""
    cols = max(1, min(max_tiles, math.ceil(width / max_side)))
    rows = max(1, min(max_tiles, math.ceil(height / max_side)))
    tile_w = math.ceil(width / cols)
    tile_h = math.ceil(height / rows)
    tiles: list[Rect] = []
    for row in range(rows):
        for col in range(cols):
            x, y = col * tile_w, row * tile_h
            tiles.append(
                Rect(x, y, min(width - x, tile_w + overlap), min(height - y, tile_h + overlap))
            )
    return tiles


def merge_boxes(boxes: list[Box], threshold: float = 0.5) -> list[Box]:
    """Drop duplicates produced by the tile overlap, keeping the better score."""
    kept: list[Box] = []
    for box in boxes:
        for index, existing in enumerate(kept):
            if iou(existing.rect, box.rect) >= threshold:
                if box.confidence > existing.confidence:
                    kept[index] = box
                break
        else:
            kept.append(box)
    return kept
