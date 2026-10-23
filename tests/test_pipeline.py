import threading
import time

import numpy as np

from ocrtrans.capture import Health
from ocrtrans.geometry import Box, Rect
from ocrtrans.pipeline import EventKind, Pipeline
from ocrtrans.track import BlockState, Decision, TemporalTracker, TrackBlock
from ocrtrans.translate import TranslationCache, TranslationService


class FakeCapture:
    def __init__(self, frame):
        self.frame = frame

    def grab(self, rect):
        return self.frame, Health.OK


class FakeOcr:
    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else [Box(Rect(0, 0, 50, 12), "X", 0.9)]

    def read(self, frame, rects=None):
        return list(self.boxes)


class FakeTracker:
    def __init__(self, crop=False, skip=False):
        self.crop = crop
        self.skip = skip

    def update(self, frame, now):
        if self.skip:
            return Decision(skip=True)
        if self.crop:
            return Decision(crops=[(1, Rect(0, 0, 50, 12))])
        return Decision(full_detect=True)

    def observe(self, boxes, now, scope=None):
        if not boxes:
            return []
        return [TrackBlock(id=1, rect=Rect(0, 0, 50, 12), text="X", state=BlockState.STABLE)]

    def blocks(self):
        return [TrackBlock(id=1, rect=Rect(0, 0, 50, 12), text="X", state=BlockState.STABLE)]

    def reset(self):
        pass


class FastEngine:
    name, kind = "fast", "local"

    def __init__(self, prefix="FAST: "):
        self.prefix = prefix

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return [self.prefix + t for t in texts]


class SlowEngine:
    kind = "online"

    def __init__(self, delay=None, name="slow", prefix="SLOW: "):
        self.name = name
        self.prefix = prefix
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.unloaded_before_finish = False
        self.delay = delay

    def load(self):
        pass

    def unload(self):
        self.unloaded_before_finish = not self.finished.is_set()

    def translate(self, texts):
        self.started.set()
        if self.delay is not None:
            time.sleep(self.delay)
        else:
            self.release.wait(5.0)
        self.finished.set()
        return [self.prefix + t for t in texts]


def pipeline_with(
    engine=None,
    ocr=None,
    tracker=None,
    tracker_returns_crop=False,
    tracker_returns_skip=False,
    region=None,
):
    events = []
    engine = engine or FastEngine()
    p = Pipeline(
        capture=FakeCapture(np.full((100, 200, 3), 30, np.uint8)),
        ocr=ocr or FakeOcr(),
        tracker=tracker
        or FakeTracker(crop=tracker_returns_crop, skip=tracker_returns_skip),
        service=TranslationService(engine, TranslationCache()),
        emit=events.append,
        fps=60,
    )
    p.set_regions([region or Rect(0, 0, 200, 100)])
    p.events = lambda: events

    def wait_event(pred, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for event in events:
                if pred(event):
                    return event
            time.sleep(0.01)
        return None

    p.wait_event = wait_event
    return p


def test_slow_online_request_does_not_block_local_ocr():
    slow = SlowEngine(delay=3.0)
    p = pipeline_with(engine=slow, ocr=FakeOcr())
    p.start()
    time.sleep(0.3)
    assert p.stats()["ocr_passes"] >= 2
    p.stop()


def test_engine_switch_drops_inflight_result():
    slow = SlowEngine(delay=None, name="slow", prefix="SLOW: ")
    p = pipeline_with(engine=slow)
    p.start()
    assert slow.started.wait(2.0), "slow inference never started"
    p.set_engine(FastEngine(prefix="FAST: "))
    slow.release.set()
    assert slow.finished.wait(2.0), "old worker never finished"
    assert slow.unloaded_before_finish is False, "engine unloaded while in use"
    emitted = [
        i.text
        for e in p.events()
        if e.kind == EventKind.TRANSLATION_RESULT
        for i in e.items
    ]
    assert not any(t.startswith("SLOW: ") for t in emitted)
    p.stop()


def test_crop_scan_reports_only_observed_ids():
    p = pipeline_with(ocr=FakeOcr(boxes=[Box(Rect(0, 0, 50, 12), "X", 0.9)]), tracker_returns_crop=True)
    p.start()
    ev = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.observed_block_ids)
    assert ev.observed_block_ids == [ev.blocks[0].block_id]
    p.stop()


def test_skip_is_reported_as_skip_not_empty_ocr():
    p = pipeline_with(tracker_returns_skip=True)
    p.start()
    assert p.wait_event(lambda e: e.kind == EventKind.SKIP) is not None
    p.stop()


def test_empty_ocr_result_carries_no_blocks():
    p = pipeline_with(ocr=FakeOcr(boxes=[]))
    p.start()
    ev = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT)
    assert ev.blocks == []
    p.stop()


def test_epoch_cancel_clears_queue():
    p = pipeline_with(engine=SlowEngine(delay=2.0))
    p.start()
    p.set_enabled(False)
    assert p.stats()["mt_queue"] == 0
    p.stop()


def test_emitted_rects_are_screen_coordinates():
    p = pipeline_with(region=Rect(100, 50, 200, 100))
    p.start()
    event = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.blocks)
    assert event.blocks[0].rect == Rect(100, 50, 50, 12)
    p.stop()


def test_pipeline_groups_stacked_lines_into_one_block():
    ocr = FakeOcr(
        boxes=[
            Box(Rect(10, 10, 180, 20), "Open the gate", 0.9),
            Box(Rect(10, 32, 180, 20), "traveler.", 0.9),
        ]
    )
    tracker = TemporalTracker(profile="fast", blind_cap_s=0.2, refresh_s=0.2)
    p = pipeline_with(ocr=ocr, tracker=tracker)
    p.start()
    event = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.blocks)
    assert len(event.blocks) == 1
    assert event.blocks[0].source_text == "Open the gate traveler."
    p.stop()
