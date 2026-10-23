import numpy as np

from ocrtrans.capture import CaptureManager, Health
from ocrtrans.geometry import Rect


class FakeBackend:
    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def grab(self, rect):
        self.calls += 1
        return self.frames.pop(0) if self.frames else None

    def close(self):
        pass


def test_none_frame_is_not_an_error():
    m = CaptureManager(FakeBackend([None, np.full((4, 4, 4), 9, np.uint8)]))
    assert m.grab(Rect(0, 0, 4, 4)) == (None, Health.OK)


def test_black_frames_switch_to_fallback():
    main = FakeBackend([np.zeros((4, 4, 3), np.uint8)] * 3)
    fb = FakeBackend([np.full((4, 4, 3), 9, np.uint8)])
    m = CaptureManager(main, fallback=fb)
    for _ in range(3):
        m.grab(Rect(0, 0, 4, 4))
    frame, health = m.grab(Rect(0, 0, 4, 4))
    assert health == Health.WGC_FALLBACK and fb.calls == 1
    assert frame is not None
