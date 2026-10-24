"""Acceptance: a subtitle over a moving background, with real OCR."""

import queue
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocrtrans.blocks import BlockMemory
from ocrtrans.capture import Health
from ocrtrans.geometry import Rect
from ocrtrans.lifecycle import OverlayState
from ocrtrans.ocr import RapidOcrEngine
from ocrtrans.pipeline import Pipeline
from ocrtrans.scan import ScanBudget
from ocrtrans.translate import TranslationCache, TranslationService

WIDTH, HEIGHT = 900, 200
FONT = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 40)


def frame_with(text: str) -> np.ndarray:
    rng = np.random.default_rng()
    noise = rng.integers(0, 80, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    image = Image.fromarray(noise)
    ImageDraw.Draw(image).text((30, 80), text, font=FONT, fill=(240, 240, 240))
    return np.array(image)[:, :, ::-1].copy()


class MovingCapture:
    def __init__(self, text: str) -> None:
        self.text = text

    def grab(self, rect):
        return frame_with(self.text), Health.OK


class FakeEngine:
    name, kind = "fake", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return [f"VI {text}" for text in texts]


def pump(events, state, deadline, on_items):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            event = events.get(timeout=0.1)
        except queue.Empty:
            continue
        on_items(state.on_event(event, time.monotonic()))


def test_moving_background_subtitle_translates_and_never_shows_the_old_text():
    capture = MovingCapture("Open the gate, traveler.")
    events: queue.Queue = queue.Queue()
    ocr = RapidOcrEngine()
    ocr.read(frame_with("warm up"))  # model load is a one-time startup cost
    pipeline = Pipeline(
        capture,
        ocr,
        TranslationService(FakeEngine(), TranslationCache()),
        events.put,
        memory=BlockMemory("fast"),
        budget=ScanBudget(15),
        fps=15,
    )
    pipeline.set_regions([Rect(0, 0, WIDTH, HEIGHT)])
    state = OverlayState()
    seen = []
    pipeline.start()
    try:
        pump(
            events,
            state,
            8.0,
            lambda items: seen.append(items),
        )
        old = [
            item
            for items in seen
            for item in items
            if "traveler" in item.source_text.casefold()
        ]
        assert old, "real OCR never read the subtitle on the moving background"
        old_source = old[-1].source_text
        old_block_id = old[-1].block_id
        old_text = old[-1].text

        capture.text = "The gate is locked."
        after = []
        end = time.monotonic() + 8.0
        new_started = None
        while time.monotonic() < end:
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                continue
            items = state.on_event(event, time.monotonic())
            after.append(items)
            if new_started is None and any("locked" in i.source_text.casefold() for i in items):
                new_started = len(after) - 1
                break
        assert new_started is not None, "the new subtitle was never translated"

        pump(events, state, 2.0, lambda items: after.append(items))
        tail = [item for items in after[new_started:] for item in items]
        assert all("traveler" not in item.source_text.casefold() for item in tail)
        assert all(
            not (
                item.block_id == old_block_id
                and item.source_text.casefold() == old_source.casefold()
                and item.text == old_text
            )
            for item in tail
        )
    finally:
        pipeline.stop()
