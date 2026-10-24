import threading
import time

import numpy as np

from ocrtrans.blocks import BlockMemory
from ocrtrans.capture import Health
from ocrtrans.geometry import Box, Rect
from ocrtrans.pipeline import EventKind, Pipeline
from ocrtrans.scan import ScanBudget
from ocrtrans.translate import TranslationCache, TranslationService


class FakeCapture:
    def __init__(self, frame):
        self.frame = frame

    def grab(self, rect):
        return self.frame, Health.OK


class NoiseCapture:
    def grab(self, rect):
        rng = np.random.default_rng()
        return rng.integers(0, 255, (100, 200, 3), dtype=np.uint8), Health.OK


class FallbackCapture:
    def grab(self, rect):
        return np.full((100, 200, 3), 30, np.uint8), Health.WGC_FALLBACK


class BlackCapture:
    def grab(self, rect):
        return None, Health.BLACK


class CountingCapture(FakeCapture):
    def __init__(self, frame):
        super().__init__(frame)
        self.grabs = 0

    def grab(self, rect):
        self.grabs += 1
        return super().grab(rect)


class FakeOcr:
    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else [Box(Rect(0, 0, 50, 12), "XXX", 0.9)]

    def read(self, frame):
        return list(self.boxes)


class BlockingOcr:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def read(self, frame):
        self.started.set()
        self.release.wait(5.0)
        return [Box(Rect(0, 0, 50, 12), "XXX", 0.9)]


class SlowOcr:
    def __init__(self, delay):
        self.delay = delay

    def read(self, frame):
        time.sleep(self.delay)
        return [Box(Rect(0, 0, 50, 12), "XXX", 0.9)]


class RaisingOcr:
    def read(self, frame):
        raise RuntimeError("ocr broke")


class WidthOcr:
    """Simulates a wide region whose small text is lost at the 1280 cap."""

    def read(self, frame):
        if frame.shape[1] >= 2560:
            return []
        return [Box(Rect(10, 10, 300, 40), "tile text", 0.9)]


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


class EmptyEngine:
    name, kind = "empty", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return ["" for _ in texts]


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
    capture=None,
    region=None,
    fps=60,
    memory=None,
):
    events = []
    p = Pipeline(
        capture=capture or FakeCapture(np.full((100, 200, 3), 30, np.uint8)),
        ocr=ocr or FakeOcr(),
        service=TranslationService(engine or FastEngine(), TranslationCache()),
        emit=events.append,
        memory=memory or BlockMemory("fast"),
        budget=ScanBudget(fps),
        fps=fps,
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
    p = pipeline_with(engine=slow)
    p.start()
    time.sleep(0.8)
    assert p.stats()["ocr_passes"] >= 2
    p.stop()


def test_engine_switch_refuses_while_a_translation_is_running():
    slow = SlowEngine(delay=None, name="slow", prefix="SLOW: ")
    p = pipeline_with(engine=slow)
    p.start()
    assert slow.started.wait(2.0), "slow inference never started"
    p.set_enabled(False)
    assert p.unload_engine(timeout=0.1) is False
    assert slow.unloaded_before_finish is False, "engine unloaded while in use"
    slow.release.set()
    assert slow.finished.wait(2.0), "old worker never finished"
    assert p.unload_engine(timeout=2.0) is True
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


def test_a_new_block_is_translated_before_it_is_stable():
    # A caption change must not leave the cover blank for a whole scan.
    memory = BlockMemory("safe")
    memory.stable_after_s = 3600.0  # never stable during this test
    p = pipeline_with(memory=memory)
    p.start()
    event = p.wait_event(lambda e: e.kind == EventKind.TRANSLATION_RESULT)
    p.stop()
    assert event is not None
    assert event.items[0].source_text == "XXX"


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
    p = pipeline_with(ocr=ocr)
    p.start()
    event = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.blocks)
    assert len(event.blocks) == 1
    assert event.blocks[0].source_text == "Open the gate traveler."
    p.stop()


def _far_boxes(count):
    return [
        Box(Rect(10, 10 + index * 40, 100, 20), f"line {index}", 0.9)
        for index in range(count)
    ]


def test_pipeline_emits_one_event_for_a_batch():
    p = pipeline_with(ocr=FakeOcr(boxes=_far_boxes(3)))
    p.start()
    event = p.wait_event(
        lambda e: e.kind == EventKind.TRANSLATION_RESULT and len(e.items) == 3
    )
    assert event is not None
    p.stop()


def test_cap_defers_blocks_instead_of_dropping_them():
    p = pipeline_with(ocr=FakeOcr(boxes=_far_boxes(20)))
    p.start()
    seen = set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and len(seen) < 20:
        time.sleep(0.05)
        for event in p.events():
            if event.kind is EventKind.TRANSLATION_RESULT:
                seen.update(item.block_id for item in event.items)
    p.stop()
    assert len(seen) == 20


def test_moving_background_does_not_increase_the_scan_rate():
    p = pipeline_with(capture=NoiseCapture(), fps=60)
    p.start()
    time.sleep(1.2)
    passes = p.stats()["ocr_passes"]
    p.stop()
    assert 2 <= passes <= 6


def test_capture_does_not_grab_while_ocr_is_running():
    cap, ocr = CountingCapture(np.full((100, 200, 3), 30, np.uint8)), BlockingOcr()
    p = pipeline_with(capture=cap, ocr=ocr, fps=60)
    p.start()
    assert ocr.started.wait(2.0)
    time.sleep(0.05)  # let any in-flight grab land
    grabs = cap.grabs
    time.sleep(0.15)
    assert cap.grabs == grabs
    ocr.release.set()
    p.stop()


def test_capture_grabs_only_when_a_pass_is_due():
    cap = CountingCapture(np.full((100, 200, 3), 30, np.uint8))
    p = pipeline_with(capture=cap, fps=60)
    p.start()
    time.sleep(1.2)
    passes = p.stats()["ocr_passes"]
    grabs = cap.grabs
    p.stop()
    assert passes >= 3
    assert grabs <= passes + 2


def test_empty_backoff_does_not_spin_capture():
    cap = CountingCapture(np.full((100, 200, 3), 30, np.uint8))
    p = pipeline_with(capture=cap, ocr=FakeOcr(boxes=[]), fps=60)
    p.start()
    time.sleep(3.0)
    passes = p.stats()["ocr_passes"]
    grabs = cap.grabs
    p.stop()
    assert passes >= 3
    assert grabs <= 20, "the capture loop spun while the empty backoff was active"


def test_scan_interval_respects_measured_ocr_time():
    p = pipeline_with(ocr=SlowOcr(0.30), fps=60)
    p.start()
    time.sleep(1.4)
    passes = p.stats()["ocr_passes"]
    p.stop()
    assert passes <= 3


def test_empty_results_back_off_to_one_pass_per_second():
    p = pipeline_with(ocr=FakeOcr(boxes=[]), fps=60)
    p.start()
    time.sleep(2.2)
    passes = p.stats()["ocr_passes"]
    p.stop()
    assert 2 <= passes <= 5


def test_a_miss_right_after_text_does_not_back_off():
    p = pipeline_with()
    p._empty_count = 5
    p._last_ocr_ms = 100.0
    p._last_text_at = p.clock() - 1.0
    assert p._scan_interval() <= 0.5


def test_a_long_empty_stretch_still_backs_off():
    p = pipeline_with()
    p._empty_count = 5
    p._last_ocr_ms = 100.0
    p._last_text_at = p.clock() - 10.0
    assert p._scan_interval() == 1.0


class SwitchCapture:
    def __init__(self):
        self.frame = np.full((100, 200, 3), 30, np.uint8)

    def grab(self, rect):
        return self.frame, Health.OK


def test_capture_health_reaches_the_overlay_when_there_is_no_frame():
    p = pipeline_with(capture=BlackCapture(), fps=60)
    p.start()
    ev = p.wait_event(
        lambda e: e.kind is EventKind.CAPTURE_HEALTH and e.health is Health.BLACK
    )
    p.stop()
    assert ev is not None, "a black capture never told the overlay to hold"


def test_a_one_tick_change_wakes_the_empty_backoff():
    cap = SwitchCapture()
    p = pipeline_with(capture=cap, ocr=FakeOcr(boxes=[]), fps=60)
    p.start()
    try:
        end = time.monotonic() + 6.0
        while time.monotonic() < end and p.stats()["ocr_passes"] < 4:
            time.sleep(0.02)
        assert p.stats()["ocr_passes"] >= 4, "the empty screen never backed off"

        before = p.stats()["ocr_passes"]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and p.stats()["ocr_passes"] == before:
            time.sleep(0.01)
        after_pass = p.stats()["ocr_passes"]
        time.sleep(0.05)  # the change must land after the floor and inside the backoff
        cap.frame = np.full((100, 200, 3), 200, np.uint8)

        end = time.monotonic() + 0.6
        while time.monotonic() < end and p.stats()["ocr_passes"] == after_pass:
            time.sleep(0.01)
        assert p.stats()["ocr_passes"] > after_pass, "the wake never shortened the wait"
    finally:
        p.stop()


def test_fallback_capture_still_runs_ocr():
    p = pipeline_with(
        capture=FallbackCapture(),
        ocr=FakeOcr(boxes=[Box(Rect(0, 0, 50, 12), "XXX", 0.9)]),
        fps=60,
    )
    p.start()
    ev = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.blocks)
    p.stop()
    assert ev is not None and ev.blocks[0].source_text == "XXX"


def test_tiled_rescue_finds_text_a_single_pass_missed():
    p = pipeline_with(
        capture=FakeCapture(np.zeros((1440, 2560, 3), np.uint8)),
        ocr=WidthOcr(),
        region=Rect(0, 0, 2560, 1440),
        fps=60,
    )
    p.start()
    ev = p.wait_event(
        lambda e: e.kind == EventKind.OCR_RESULT
        and any(b.source_text == "tile text" for b in e.blocks),
        timeout=4.0,
    )
    p.stop()
    assert ev is not None


def test_scan_ids_increase_and_reset_with_the_epoch():
    p = pipeline_with(fps=60)
    p.start()
    p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT)
    first = [e.scan_id for e in p.events() if e.kind == EventKind.OCR_RESULT]
    before = len(p.events())
    p.set_regions([Rect(0, 0, 200, 100)])
    time.sleep(0.6)
    after = p.events()[before:]
    p.stop()
    assert first == sorted(set(first)) and first[0] == 1
    assert any(e.kind is EventKind.RESET for e in after)
    ids = [e.scan_id for e in after if e.kind == EventKind.OCR_RESULT]
    assert ids and ids[0] == 1


def test_empty_translation_keeps_the_block_hidden():
    p = pipeline_with(engine=EmptyEngine(), fps=60)
    p.start()
    p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.blocks)
    time.sleep(0.3)
    p.stop()
    assert p.stats()["mt_errors"] == 0
    assert not [e for e in p.events() if e.kind is EventKind.TRANSLATION_RESULT]


class LifecycleEngine:
    name, kind = "lifecycle", "local"

    def __init__(self):
        self.loads = 0
        self.unloads = 0

    def load(self):
        self.loads += 1

    def unload(self):
        self.unloads += 1

    def translate(self, texts):
        return list(texts)


def test_adopt_engine_frees_the_old_model_without_reloading():
    old = LifecycleEngine()
    new = LifecycleEngine()
    p = pipeline_with(engine=old)
    old.load()
    new.load()
    p.set_enabled(False)
    assert p.unload_engine(timeout=0.1) is True
    p.adopt_engine(new)
    assert old.unloads == 1
    assert new.loads == 1
    p.stop()


def test_ocr_failure_is_counted_not_silent():
    p = pipeline_with(ocr=RaisingOcr())
    p.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and p.stats()["ocr_errors"] == 0:
        time.sleep(0.05)
    stats = p.stats()
    p.stop()
    assert stats["ocr_errors"] >= 1
    assert stats["ocr_passes"] == 0


def test_stats_report_how_many_blocks_the_last_pass_found():
    p = pipeline_with()
    p.start()
    assert p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.blocks)
    assert p.stats()["blocks"] == 1
    p.stop()


def test_stats_report_scan_rate_and_last_pass_duration():
    p = pipeline_with(fps=60)
    p.start()
    assert p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT)
    time.sleep(0.3)
    stats = p.stats()
    p.stop()
    assert stats["scan_rate"] > 0
    assert stats["last_scan_ms"] > 0
    assert "tiled" in stats


class BadEngine:
    name, kind = "bad", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        raise RuntimeError("translator broke")


def test_translation_failure_is_counted_not_silent():
    p = pipeline_with(engine=BadEngine())
    p.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and p.stats()["mt_errors"] == 0:
        time.sleep(0.05)
    stats = p.stats()
    p.stop()
    assert stats["mt_errors"] >= 1


class CountingOcr(FakeOcr):
    def __init__(self, boxes=None):
        super().__init__(boxes)
        self.calls = 0

    def read(self, frame):
        self.calls += 1
        return super().read(frame)


def test_a_thin_strip_never_runs_the_tiled_rescue():
    ocr = CountingOcr(boxes=[])
    p = pipeline_with(
        capture=FakeCapture(np.zeros((53, 1760, 3), np.uint8)),
        ocr=ocr,
        region=Rect(0, 0, 1760, 53),
        fps=60,
    )
    p.start()
    time.sleep(1.2)
    stats = p.stats()
    p.stop()
    assert stats["ocr_passes"] >= 2
    assert stats["tiled"] == 0, "a wide subtitle strip was split into tiles"
    assert ocr.calls == stats["ocr_passes"], "a miss re-read the strip in tiles"


def test_a_stable_text_set_scans_at_half_rate():
    p = pipeline_with(fps=60)
    p.start()
    time.sleep(3.0)
    passes = p.stats()["ocr_passes"]
    p.stop()
    assert passes >= 5
    assert passes <= 9, "a static snapshot kept the full 4 passes/s"
