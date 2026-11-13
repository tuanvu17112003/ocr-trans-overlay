"""Stability regressions: the numbers the user reported as "it flickers".

These run the real pipeline with a fake capture and real OCR, then assert the
measured pass rate and cover stability. They are slow (a few seconds each) but
they are the only tests that catch the interaction between scheduling, the
candidate gate and the overlay state machine.
"""

import queue
import time

import numpy as np

from ocrtrans.blocks import BlockMemory
from ocrtrans.capture import Health
from ocrtrans.geometry import Rect
from ocrtrans.lifecycle import OverlayState
from ocrtrans.ocr import RapidOcrEngine
from ocrtrans.pipeline import Pipeline
from ocrtrans.scan import ScanBudget
from ocrtrans.translate import TranslationCache, TranslationService

WIDTH, HEIGHT = 900, 200


class FakeEngine:
    name, kind = "fake", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return ["VI " + text for text in texts]


def caption_frame(text: str, seed: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 40)
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 80, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    image = Image.fromarray(noise)
    ImageDraw.Draw(image).text((30, 80), text, font=font, fill=(240, 240, 240))
    return np.array(image)[:, :, ::-1].copy()


class WobbleCapture:
    """The caption never changes; only the background under it does."""

    def __init__(self, text: str) -> None:
        self.frames = [caption_frame(text, seed) for seed in range(6)]
        self.index = 0

    def grab(self, rect):
        frame = self.frames[self.index % len(self.frames)]
        self.index += 1
        return frame, Health.OK


def run(capture, seconds: float, fps: int = 15):
    events: queue.Queue = queue.Queue()
    ocr = RapidOcrEngine()
    ocr.read(caption_frame("warm up", 0))  # pay the model load outside the window
    pipeline = Pipeline(
        capture,
        ocr,
        TranslationService(FakeEngine(), TranslationCache()),
        events.put,
        memory=BlockMemory("fast"),
        budget=ScanBudget(fps),
        fps=fps,
    )
    pipeline.set_regions([Rect(0, 0, WIDTH, HEIGHT)])
    state = OverlayState()
    transitions = 0
    shown_before = False
    requests = 0
    pipeline.start()
    started = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.kind.name == "TRANSLATION_RESULT":
                requests += len(event.items)
            items = state.on_event(event, time.monotonic())
            if bool(items) != shown_before:
                transitions += 1
                shown_before = bool(items)
        passes = pipeline.stats()["ocr_passes"]
    finally:
        pipeline.stop()
    return passes, transitions, requests


def test_a_moving_background_does_not_strobe_a_static_caption():
    """Reported symptom 1: the background moves, the text does not, the overlay blinks."""
    passes, transitions, requests = run(WobbleCapture("Open the gate, traveler."), 6.0)
    assert transitions <= 1, f"the cover strobed {transitions} times on a static caption"
    assert requests == 1, f"one sentence caused {requests} translation requests"
    # Before the fix this measured 3.96 pass/s; the cap here is deliberately loose
    # enough for a slow CI machine while still failing the old behaviour.
    assert passes / 6.0 <= 2.5, f"scan rate was {passes / 6.0:.2f}/s for a static caption"


def test_a_real_caption_change_is_still_handled_quickly():
    """Stability must not come at the cost of ignoring a genuine change."""

    class SwitchCapture:
        def __init__(self) -> None:
            self.text = "Open the gate, traveler."
            self.frames = 0

        def grab(self, rect):
            self.frames += 1
            if self.frames == 12:
                self.text = "The gate is locked."
            return caption_frame(self.text, self.frames), Health.OK

    passes, transitions, requests = run(SwitchCapture(), 8.0)
    assert requests >= 2, "the second caption was never translated"
    assert passes >= 3, "the scanner stopped after the caption changed"
