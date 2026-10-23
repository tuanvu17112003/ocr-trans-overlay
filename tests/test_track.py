import numpy as np

from ocrtrans.geometry import Box, Rect
from ocrtrans.track import BlockState, TemporalTracker


def test_animated_background_does_not_ocr_every_frame():
    t = TemporalTracker()
    base = np.full((100, 200), 30, np.uint8)
    for _ in range(3):
        t.observe([Box(Rect(20, 20, 160, 30), "HELLO THERE", 0.99)], 0.0)
    calls = 0
    for i in range(10):
        frame = base.copy()
        frame[70:100, :] = (i * 20) % 255
        d = t.update(frame, 0.1 * i)
        calls += d.full_detect or bool(d.crops)
    assert 1 <= calls <= 3


def test_new_text_under_activity_detected_within_throttle():
    t = TemporalTracker()
    for _ in range(3):
        t.observe([Box(Rect(20, 20, 160, 30), "OLD", 0.99)], 0.0)
    base = np.full((100, 200), 30, np.uint8)
    detections = 0
    for i in range(6):
        frame = base.copy()
        frame[70:100, :] = (i * 30) % 255
        detections += t.update(frame, 0.067 * i).full_detect
    assert 1 <= detections <= 2


def test_blind_cap_fires_within_1_5s_of_static_screen():
    t = TemporalTracker(blind_cap_s=1.5)
    frame = np.full((100, 200), 30, np.uint8)
    for _ in range(3):
        t.observe([Box(Rect(10, 10, 100, 20), "HELLO", 0.9)], 0.0)
    assert t.blocks()[0].state is BlockState.STABLE
    t.update(frame, 0.0)
    first_detect_at = None
    for i in range(1, 80):
        now = 0.05 * i
        if t.update(frame, now).full_detect:
            first_detect_at = now
            break
    assert first_detect_at is not None
    assert 1.4 <= first_detect_at <= 1.5


def test_scene_cut_resets_blocks():
    t = TemporalTracker()
    t.update(np.full((100, 200), 30, np.uint8), 0.0)
    t.observe([Box(Rect(10, 10, 100, 20), "A", 0.9)], 0.0)
    d = t.update(np.full((100, 200), 255, np.uint8), 0.1)
    assert d.full_detect is True and t.blocks() == []


def test_growth_resets_stability():
    t = TemporalTracker(profile="safe")
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO", 0.9)], 0.0)
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO THER", 0.9)], 0.07)
    assert t.blocks()[0].state == BlockState.PENDING
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO THERE", 0.9)], 0.14)
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO THERE", 0.9)], 0.21)
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO THERE", 0.9)], 0.28)
    assert t.blocks()[0].state == BlockState.STABLE


def test_set_profile_changes_stability_window():
    t = TemporalTracker(profile="safe")
    t.set_profile("fast")
    box = Box(Rect(10, 10, 100, 20), "A", 0.9)
    t.observe([box], 0.0)
    t.observe([box], 0.07)
    assert t.blocks()[0].state is BlockState.STABLE



def test_block_keeps_line_height():
    t = TemporalTracker()
    t.observe([Box(Rect(0, 0, 200, 20), "A", 0.9, line_px=18)], 0.0)
    assert t.blocks()[0].line_px == 18


def test_grown_paragraph_keeps_its_id():
    t = TemporalTracker()
    t.observe([Box(Rect(0, 0, 200, 20), "A", 0.9)], 0.0)
    first = t.blocks()[0].id
    t.observe([Box(Rect(0, 0, 200, 60), "A B C", 0.9)], 0.07)
    blocks = t.blocks()
    assert len(blocks) == 1
    assert blocks[0].id == first



def test_two_boxes_do_not_share_one_block():
    t = TemporalTracker()
    t.observe([Box(Rect(0, 0, 200, 60), "A B", 0.9)], 0.0)
    t.observe([Box(Rect(0, 0, 200, 20), "A", 0.9), Box(Rect(0, 40, 200, 20), "B", 0.9)], 0.07)
    assert len(t.blocks()) == 2


def test_static_screen_without_text_skips_between_full_detects():
    t = TemporalTracker("safe")
    frame = np.zeros((100, 120, 3), np.uint8)
    assert t.update(frame, 0.0).full_detect
    t.observe([], 0.0)
    skipped = [t.update(frame, 0.06 * index) for index in range(1, 8)]
    assert all(decision.skip for decision in skipped)
    assert t.update(frame, 1.5).full_detect
