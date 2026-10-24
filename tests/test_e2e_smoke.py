import queue
import time

import numpy as np

from ocrtrans.blocks import BlockMemory
from ocrtrans.capture import Health
from ocrtrans.geometry import Box, Rect
from ocrtrans.lifecycle import OverlayState
from ocrtrans.pipeline import Pipeline
from ocrtrans.scan import ScanBudget
from ocrtrans.translate import TranslationCache, TranslationService


class FakeCapture:
    def grab(self, rect):
        return np.full((100, 200, 3), 30, np.uint8), Health.OK


class FakeOcr:
    def __init__(self, boxes):
        self.boxes = boxes

    def read(self, frame):
        return list(self.boxes)


class FakeEngine:
    name, kind = "fake", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return [t.upper() for t in texts]


def pump_until(events, state, pred, timeout=3.0):
    end = time.monotonic() + timeout
    items = []
    while time.monotonic() < end:
        try:
            event = events.get(timeout=0.1)
        except queue.Empty:
            continue
        items = state.on_event(event, time.monotonic())
        if pred(items):
            return True
    return pred(items)


def test_pipeline_emits_translation_then_clears_after_the_text_disappears():
    events: queue.Queue = queue.Queue()
    ocr = FakeOcr([Box(Rect(10, 10, 100, 20), "Open the gate", 0.9)])
    pipeline = Pipeline(
        FakeCapture(),
        ocr,
        TranslationService(FakeEngine(), TranslationCache()),
        events.put,
        memory=BlockMemory("fast"),
        budget=ScanBudget(60),
        fps=60,
    )
    pipeline.set_regions([Rect(0, 0, 200, 100)])
    state = OverlayState()
    pipeline.start()
    try:
        assert pump_until(events, state, lambda items: bool(items)), (
            "the subtitle was never translated and rendered"
        )
        ocr.boxes = []
        assert pump_until(events, state, lambda items: not items), (
            "the old translation stayed after the text disappeared"
        )
    finally:
        pipeline.stop()
