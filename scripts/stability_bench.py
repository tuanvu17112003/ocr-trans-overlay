"""Measure the stability properties this branch changed.

Runs the real pipeline with a fake capture and synthetic frames, so the numbers
are comparable across runs without a game open. Prints one markdown table.

    .venv\\Scripts\\python.exe scripts\\stability_bench.py
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ocrtrans.blocks import BlockMemory  # noqa: E402
from ocrtrans.capture import Health  # noqa: E402
from ocrtrans.candidates import GateStats, filter_candidates  # noqa: E402
from ocrtrans.geometry import Box, Rect  # noqa: E402
from ocrtrans.lifecycle import OverlayState  # noqa: E402
from ocrtrans.ocr import RapidOcrEngine  # noqa: E402
from ocrtrans.pipeline import Pipeline  # noqa: E402
from ocrtrans.scan import ScanBudget  # noqa: E402
from ocrtrans.translate import TranslationCache, TranslationService  # noqa: E402

WIDTH, HEIGHT = 900, 200
FONT = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 40)


class FakeCapture:
    def __init__(self, frame):
        self.frame = frame

    def grab(self, rect):
        return self.frame, Health.OK


class FakeEngine:
    name, kind = "fake", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return ["VI " + text for text in texts]


def caption_frame(text: str, background_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(background_seed)
    noise = rng.integers(0, 80, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    image = Image.fromarray(noise)
    ImageDraw.Draw(image).text((30, 80), text, font=FONT, fill=(240, 240, 240))
    return np.array(image)[:, :, ::-1].copy()


class WobbleCapture:
    """A static caption whose background changes, which makes OCR wobble."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.frames = [caption_frame(text, seed) for seed in range(6)]

    def grab(self, rect):
        self.calls += 1
        return self.frames[self.calls % len(self.frames)], Health.OK


def run_pipeline(capture, seconds: float, fps: int = 15):
    events: queue.Queue = queue.Queue()
    engine = RapidOcrEngine()
    pipeline = Pipeline(
        capture,
        engine,
        TranslationService(FakeEngine(), TranslationCache()),
        events.put,
        memory=BlockMemory("fast"),
        budget=ScanBudget(fps),
        fps=fps,
    )
    pipeline.set_regions([Rect(0, 0, WIDTH, HEIGHT)])
    state = OverlayState()
    transitions = 0
    was_shown = False
    mt_requests = 0
    pipeline.start()
    started = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.kind.name == "TRANSLATION_RESULT":
                mt_requests += len(event.items)
            items = state.on_event(event, time.monotonic())
            shown = bool(items)
            if shown != was_shown:
                transitions += 1
                was_shown = shown
        passes = pipeline.stats()["ocr_passes"]
    finally:
        pipeline.stop()
    return {
        "passes": passes,
        "pass_rate": round(passes / seconds, 2),
        "cover_transitions": transitions,
        "mt_requests": mt_requests,
    }


def gate_numbers():
    """Artwork rejection on the synthetic icon grid used by the tests."""
    image = Image.new("RGB", (900, 400), (34, 30, 28))
    draw = ImageDraw.Draw(image)
    for row in range(3):
        for col in range(6):
            x, y = 30 + col * 140, 30 + row * 120
            draw.rounded_rectangle((x, y, x + 110, y + 100), radius=12, fill=(70, 60, 55),
                                   outline=(150, 140, 130), width=3)
            draw.ellipse((x + 30, y + 20, x + 80, y + 70), fill=(200, 190, 170))
            draw.line((x + 20, y + 80, x + 90, y + 80), fill=(180, 170, 160), width=5)
            draw.rectangle((x + 40, y + 40, x + 70, y + 55), fill=(40, 35, 30))
    frame = np.array(image)[:, :, ::-1].copy()
    engine = RapidOcrEngine()
    raw = engine.read(frame)
    stats = GateStats()
    kept = filter_candidates(raw, frame, profile="subtitle", region=Rect(0, 0, 900, 400), stats=stats)
    caption = caption_frame("Open the gate, traveler.")
    raw_caption = engine.read(caption)
    kept_caption = filter_candidates(
        raw_caption, caption, profile="subtitle", region=Rect(0, 0, WIDTH, HEIGHT)
    )
    return {
        "art_raw": len(raw),
        "art_kept": len(kept),
        "caption_raw": len(raw_caption),
        "caption_kept": len(kept_caption),
        "reasons": stats.as_dict(),
    }


def main() -> int:
    print("## Measured on this machine\n")
    print("| scenario | passes | pass/s | cover transitions | translation requests |")
    print("|---|---|---|---|---|")
    static = run_pipeline(FakeCapture(caption_frame("Open the gate, traveler.", 0)), 4.0)
    print(
        f"| static caption, static background | {static['passes']} | {static['pass_rate']} "
        f"| {static['cover_transitions']} | {static['mt_requests']} |"
    )
    moving = run_pipeline(WobbleCapture("Open the gate, traveler."), 6.0)
    print(
        f"| static caption, moving background | {moving['passes']} | {moving['pass_rate']} "
        f"| {moving['cover_transitions']} | {moving['mt_requests']} |"
    )
    gate = gate_numbers()
    print()
    print("| candidate gate | raw boxes | kept |")
    print("|---|---|---|")
    print(f"| icon grid, no text | {gate['art_raw']} | {gate['art_kept']} |")
    print(f"| real caption | {gate['caption_raw']} | {gate['caption_kept']} |")
    print()
    print(f"rejection reasons: {gate['reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
