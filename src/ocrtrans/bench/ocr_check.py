"""Render corpus lines and measure OCR recovery."""

from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_line(text: str, path: Path, size: tuple[int, int] = (800, 120), font_px: int = 36) -> Path:
    image = Image.new("RGB", size, (18, 18, 22))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", font_px)
    draw.text((20, (size[1] - font_px) // 2), text, font=font, fill=(240, 240, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def frame_from_line(text: str) -> np.ndarray:
    image = Image.new("RGB", (800, 120), (18, 18, 22))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 36)
    draw.text((20, 40), text, font=font, fill=(240, 240, 240))
    return np.array(image)[:, :, ::-1].copy()


def check_ocr(engine, texts: list[str]) -> dict:
    ratios: list[float] = []
    for text in texts:
        frame = frame_from_line(text)
        boxes = engine.read(frame)
        got = " ".join(b.text for b in boxes).strip()
        ratios.append(difflib.SequenceMatcher(None, got.lower(), text.lower()).ratio())
    if not ratios:
        return {"mean_ratio": 0.0, "exact_rate": 0.0, "n": 0}
    return {
        "mean_ratio": round(sum(ratios) / len(ratios), 3),
        "exact_rate": round(sum(1 for r in ratios if r >= 0.95) / len(ratios), 3),
        "n": len(ratios),
    }
