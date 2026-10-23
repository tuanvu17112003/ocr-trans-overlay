import queue
import time

import numpy as np

from ocrtrans.capture import Health
from ocrtrans.geometry import Box, Rect
from ocrtrans.lifecycle import OverlayState
from ocrtrans.pipeline import EventKind, Pipeline
from ocrtrans.track import TemporalTracker
from ocrtrans.translate import TranslationCache, TranslationService


class FakeCapture:
    def grab(self, rect):
        return np.full((100, 200, 3), 30, np.uint8), Health.OK


class FakeOcr:
    def __init__(self, boxes):
        self.boxes = boxes

    def read(self, frame, rects=None):
        return list(self.boxes)


class FakeEngine:
    name, kind = "fake", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return [t.upper() for t in texts]


def test_pipeline_emits_translation_then_clears_after_two_empty_passes():
    events: queue.Queue = queue.Queue()
    ocr = FakeOcr([Box(Rect(10, 10, 100, 20), "Open the gate", 0.9)])
    pipeline = Pipeline(
        FakeCapture(),
        ocr,
        TemporalTracker(profile="fast", blind_cap_s=0.2, refresh_s=0.2),
        TranslationService(FakeEngine(), TranslationCache()),
        events.put,
        fps=60,
    )
    pipeline.set_regions([Rect(0, 0, 200, 100)])
    state = OverlayState()
    pipeline.start()

    items = []
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not items:
        try:
            event = events.get(timeout=0.1)
        except queue.Empty:
            continue
        rendered = state.on_event(event, time.monotonic())
        if event.kind is EventKind.TRANSLATION_RESULT:
            items = rendered
    assert any("OPEN THE GATE" in item.text for item in items)

    ocr.boxes = []
    empties = 0
    cleared = False
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            event = events.get(timeout=0.1)
        except queue.Empty:
            continue
        rendered = state.on_event(event, time.monotonic())
        if event.kind is EventKind.OCR_RESULT and not event.blocks:
            empties += 1
        if empties >= 2 and rendered == []:
            cleared = True
            break
    pipeline.stop()
    assert cleared
