import numpy as np

from ocrtrans.capture import CaptureManager, DxgiBackend, Health, WgcBackend
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


class FakeNativeFrame:
    color_format = "rgba16f"

    def __init__(self, image):
        self.image = image

    def to_numpy(self):
        return self.image

    def to_bgr(self):
        raise AssertionError("the full frame must not be tonemapped")


class FakeSession:
    def __init__(self, image):
        self.image = image
        self.calls = 0

    def acquire_frame(self, timeout_ms=16):
        self.calls += 1
        return FakeNativeFrame(self.image)


def test_dxgi_grab_tonemaps_only_the_region():
    image = np.full((20, 30, 4), 0.5, np.float16)
    image[2, 3] = (1.0, 0.5, 0.75, 1.0)  # rgba16f
    backend = DxgiBackend()
    backend._session = FakeSession(image)

    frame = backend.grab(Rect(1, 1, 4, 3))

    assert frame.shape == (3, 4, 3)
    assert tuple(int(value) for value in frame[1, 2]) == (191, 127, 255)
    assert backend._session.calls == 1


def test_dxgi_grab_clamps_a_region_outside_the_frame():
    image = np.full((10, 10, 4), 0.5, np.float16)
    backend = DxgiBackend()
    backend._session = FakeSession(image)
    frame = backend.grab(Rect(8, 8, 50, 50))
    assert frame.shape == (2, 2, 3)


class FakeWgcFrame:
    def __init__(self, buffer):
        self.buffer = buffer

    def convert_to_bgr(self):
        frame = type("Frame", (), {})()
        frame.frame_buffer = self.buffer[:, :, :3]
        return frame


def test_wgc_stores_only_the_region_of_each_frame():
    backend = WgcBackend.__new__(WgcBackend)
    backend._monitor_rect = None
    backend._retried_monitor = True
    import threading

    backend._lock = threading.Lock()
    backend._latest = None
    backend._rect = Rect(1, 1, 4, 3)
    backend._wanted = True
    backend._seq = 0
    backend._cv = threading.Condition(backend._lock)

    buffer = np.full((20, 30, 4), 7, np.uint8)
    buffer[2, 3] = (10, 20, 30, 255)
    backend._store_frame(FakeWgcFrame(buffer))

    assert backend._latest.shape == (3, 4, 3)
    assert tuple(backend._latest[1, 2]) == (10, 20, 30)


class DarkRegionBackend(FakeBackend):
    full_dark = False

    def grab(self, rect):
        self.calls += 1
        return np.zeros((4, 4, 3), np.uint8)


def test_a_dark_subtitle_region_on_a_bright_frame_is_not_black():
    m = CaptureManager(DarkRegionBackend([]))
    frame, health = m.grab(Rect(0, 0, 4, 4))
    assert health == Health.OK
    assert frame is not None


def test_a_black_full_frame_still_trips_the_fallback():
    main = DarkRegionBackend([None] * 5)
    main.full_dark = True
    m = CaptureManager(main, fallback=FakeBackend([np.full((4, 4, 3), 9, np.uint8)]))
    for _ in range(3):
        m.grab(Rect(0, 0, 4, 4))
    assert m.grab(Rect(0, 0, 4, 4))[1] == Health.WGC_FALLBACK


def test_dxgi_close_drops_the_session_on_the_calling_thread():
    backend = DxgiBackend()
    backend._session = object()
    backend.close()
    assert backend._session is None
    assert backend._closed is True


def test_wgc_converts_only_the_requested_frame():
    import threading

    from ocrtrans.capture import WgcBackend

    class Converted:
        frame_buffer = np.full((8, 8, 3), 200, np.uint8)

    class Frame:
        def __init__(self):
            self.converted = 0

        def convert_to_bgr(self):
            self.converted += 1
            return Converted()

    backend = WgcBackend.__new__(WgcBackend)
    backend._monitor_rect = None
    backend._retried_monitor = True
    backend._latest = None
    backend._lock = threading.Lock()
    backend._cv = threading.Condition(backend._lock)
    backend._rect = Rect(1, 1, 2, 2)
    backend._wanted = False
    backend._seq = 0
    backend.full_dark = False

    frame = Frame()
    backend._store_frame(frame)
    assert frame.converted == 0, "an unwanted frame was converted"
    backend._wanted = True
    backend._store_frame(frame)
    assert frame.converted == 1
    assert backend._latest.shape == (2, 2, 3)


def test_primary_index_follows_the_flag_not_the_position():
    from ocrtrans.capture import primary_monitor_index

    monitors = [(Rect(0, 0, 1920, 1080), False), (Rect(1920, 0, 2560, 1440), True)]
    assert primary_monitor_index(monitors) == 2


def test_primary_index_is_one_when_the_primary_is_first():
    from ocrtrans.capture import primary_monitor_index

    assert primary_monitor_index([(Rect(0, 0, 1920, 1080), True)]) == 1


def test_primary_index_falls_back_without_a_flag():
    from ocrtrans.capture import primary_monitor_index

    assert primary_monitor_index([(Rect(0, 0, 1920, 1080), False)], fallback=1) == 1
    assert primary_monitor_index([], fallback=1) == 1


def test_frame_size_must_match_the_monitor():
    from ocrtrans.capture import monitor_size_matches

    assert monitor_size_matches((1920, 1080), Rect(0, 0, 1920, 1080)) is True
    assert monitor_size_matches((2560, 1440), Rect(0, 0, 1920, 1080)) is False


def test_retry_happens_once_on_a_size_mismatch():
    from ocrtrans.capture import needs_monitor_retry

    rect = Rect(0, 0, 1920, 1080)
    assert needs_monitor_retry((2560, 1440), rect, already_retried=False) is True
    assert needs_monitor_retry((2560, 1440), rect, already_retried=True) is False
    assert needs_monitor_retry((1920, 1080), rect, already_retried=False) is False
    assert needs_monitor_retry((2560, 1440), None, already_retried=False) is False


def test_enumerate_monitors_never_raises():
    from ocrtrans.capture import enumerate_monitors

    monitors = enumerate_monitors()
    assert isinstance(monitors, list)
    for rect, primary in monitors:
        assert isinstance(rect, Rect) and isinstance(primary, bool)


def test_a_mismatching_first_frame_asks_for_a_retry(qapp):
    import threading

    backend = WgcBackend.__new__(WgcBackend)
    backend._lock = threading.Lock()
    backend._cv = threading.Condition(backend._lock)
    backend._latest = None
    backend._rect = Rect(0, 0, 4, 3)
    backend._wanted = True
    backend._seq = 0
    backend._monitor_rect = Rect(0, 0, 2560, 1440)
    backend._retried_monitor = False
    backend._retry_requested = False
    backend.full_dark = False

    backend._store_frame(FakeWgcFrame(np.zeros((1080, 1920, 4), dtype=np.uint8)))

    assert backend._retry_requested is True
    assert backend._retried_monitor is True
