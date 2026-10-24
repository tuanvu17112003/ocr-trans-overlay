"""Screen capture backends and the DXGI -> WGC watchdog."""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from enum import Enum, auto
from typing import Protocol

import cv2
import numpy as np

from .geometry import Rect

log = logging.getLogger(__name__)

MONITORINFOF_PRIMARY = 1


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", ctypes.c_ulong),
    ]


def enumerate_monitors() -> list[tuple[Rect, bool]]:
    """Every active monitor in EnumDisplayMonitors order, rects in physical px.

    The capture library indexes monitors one-based into this same enumeration,
    so the position in this list is the index to pass it. Returns [] on failure
    (the caller then keeps index 1).
    """
    monitors: list[tuple[Rect, bool]] = []
    try:
        user32 = ctypes.windll.user32
        user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MONITORINFO)]
        user32.GetMonitorInfoW.restype = ctypes.c_int
        user32.EnumDisplayMonitors.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        user32.EnumDisplayMonitors.restype = ctypes.c_int
        callback_type = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_RECT),
            ctypes.c_void_p,
        )

        def collect(hmonitor, hdc, lprc, data) -> int:
            info = _MONITORINFO()
            info.cbSize = ctypes.sizeof(_MONITORINFO)
            if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
                r = info.rcMonitor
                monitors.append(
                    (
                        Rect(r.left, r.top, r.right - r.left, r.bottom - r.top),
                        bool(info.dwFlags & MONITORINFOF_PRIMARY),
                    )
                )
            return 1

        user32.EnumDisplayMonitors(None, None, callback_type(collect), 0)
    except Exception:  # noqa: BLE001 - enumeration is best effort
        log.exception("monitor enumeration failed; assuming index 1 is primary")
        return []
    return monitors


def primary_monitor_index(monitors: list[tuple[Rect, bool]], fallback: int = 1) -> int:
    for index, (_rect, primary) in enumerate(monitors, start=1):
        if primary:
            return index
    return fallback


def monitor_size_matches(frame_wh: tuple[int, int], monitor: Rect) -> bool:
    return (frame_wh[0], frame_wh[1]) == (monitor.w, monitor.h)


def needs_monitor_retry(
    frame_wh: tuple[int, int], monitor_rect: Rect | None, already_retried: bool
) -> bool:
    if monitor_rect is None or already_retried:
        return False
    return not monitor_size_matches(frame_wh, monitor_rect)


class Health(Enum):
    OK = auto()
    RECREATING = auto()
    WGC_FALLBACK = auto()
    BLACK = auto()
    BLOCKED = auto()


class CaptureBackend(Protocol):
    def grab(self, rect: Rect) -> np.ndarray | None: ...

    def close(self) -> None: ...


def _is_dark_sample(image: np.ndarray, color_format: str) -> bool:
    """Whole-frame luma at 1/64 stride: an empty subtitle strip is legitimately dark."""
    sample = image[::64, ::64]
    if color_format == "rgba16f":
        return float(sample.astype(np.float32).mean()) < 0.01
    return float(sample.mean()) < 2.0


def _crop_bgr(image: np.ndarray, rect: Rect, color_format: str) -> np.ndarray:
    """Tonemap only the requested region; a full 1440p HDR frame costs ~50 ms."""
    height, width = image.shape[:2]
    x0 = min(max(0, rect.x), width)
    x1 = min(max(x0, rect.right), width)
    y0 = min(max(0, rect.y), height)
    y1 = min(max(y0, rect.bottom), height)
    crop = image[y0:y1, x0:x1]
    if color_format == "bgra8":
        return crop[..., :3].copy()
    if color_format == "rgba8":
        return crop[..., 2::-1].copy()
    normalized = np.clip(crop.astype(np.float32), 0.0, 1.0)
    return (normalized[..., [2, 1, 0]] * 255.0).astype(np.uint8)


class DxgiBackend:
    """Desktop Duplication. HDR screens return rgba16f, so always to_bgr()."""

    def __init__(self, monitor_index: int | None = None) -> None:
        self._monitor_index = monitor_index
        self._session = None
        self._closed = False
        self._first = True
        self.full_dark = False

    def _ensure(self):
        # DXGI sessions are thread-affine: create on the capture thread.
        if self._session is None:
            from windows_capture import DxgiDuplicationSession

            self._session = DxgiDuplicationSession(monitor_index=self._monitor_index)
        return self._session

    def grab(self, rect: Rect) -> np.ndarray | None:
        session = self._ensure()
        img = None
        # The first frames after session creation are often black; retry briefly.
        for _ in range(4 if self._first else 1):
            frame = session.acquire_frame(timeout_ms=16)
            if frame is None:
                continue
            image = frame.to_numpy()
            self.full_dark = _is_dark_sample(image, frame.color_format)
            img = _crop_bgr(image, rect, frame.color_format)
            if not self.full_dark:
                self._first = False
                break
        return img

    def recreate(self) -> None:
        if self._session is not None:
            self._session.recreate()
        self._first = True
        self.full_dark = False

    def close(self) -> None:
        # Must run on the thread that created the session (the capture thread):
        # pyo3 panics if the native session is dropped on any other thread.
        session, self._session = self._session, None
        self._closed = True
        del session


class WgcBackend:
    """Windows Graphics Capture of one window or monitor, latest frame kept."""

    def __init__(
        self,
        window_name: str | None = None,
        monitor_index: int | None = None,
        window_hwnd: int | None = None,
        monitor_rect: Rect | None = None,
    ) -> None:
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._rect: Rect | None = None
        self._wanted = False
        self._seq = 0
        self._window_name = window_name
        self._window_hwnd = window_hwnd
        self._monitor_rect = monitor_rect
        self._retried_monitor = False
        self._retry_requested = False
        self.monitor_mismatch = False
        self.full_dark = False
        self._thread: threading.Thread | None = None
        self._build_capture(monitor_index)

    def _build_capture(self, monitor_index: int | None) -> None:
        from windows_capture import WindowsCapture

        self._capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            window_name=self._window_name if self._window_hwnd is None else None,
            window_hwnd=self._window_hwnd,
            monitor_index=monitor_index if (self._window_name is None and self._window_hwnd is None) else None,
        )

        @self._capture.event
        def on_frame_arrived(frame, control) -> None:
            self._store_frame(frame)

        @self._capture.event
        def on_closed() -> None:
            pass

    def _ensure_started(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._capture.start, daemon=True)
            self._thread.start()

    def _store_frame(self, frame) -> None:
        with self._lock:
            # Convert and crop only for a requested grab: at 60 Hz a full-frame
            # BGR conversion would burn a core for nothing.
            if not self._wanted or self._rect is None:
                return
            rect = self._rect
            self._wanted = False
        image = frame.convert_to_bgr().frame_buffer
        self.full_dark = _is_dark_sample(image, "bgra8")
        height, width = image.shape[:2]
        with self._lock:
            retry = needs_monitor_retry(
                (width, height), self._monitor_rect, already_retried=self._retried_monitor
            )
            if retry:
                self._retried_monitor = True
                self._retry_requested = True
        if retry:
            log.error(
                "captured %sx%s but the primary monitor is %sx%s; retrying monitor 1",
                width,
                height,
                self._monitor_rect.w if self._monitor_rect else "?",
                self._monitor_rect.h if self._monitor_rect else "?",
            )
        x0 = min(max(0, rect.x), width)
        x1 = min(max(x0, rect.right), width)
        y0 = min(max(0, rect.y), height)
        y1 = min(max(y0, rect.bottom), height)
        crop = image[y0:y1, x0:x1].copy()
        with self._lock:
            self._latest = crop
            self._seq += 1
            self._cv.notify_all()

    def grab(self, rect: Rect) -> np.ndarray | None:
        with self._lock:
            retry = self._retry_requested
            if retry:
                self._retry_requested = False
        if retry:
            self.monitor_mismatch = True
            self._restart_with_primary()
        with self._lock:
            self._rect = rect
            self._wanted = True
            seen = self._seq
        self._ensure_started()
        with self._cv:
            self._cv.wait_for(lambda: self._seq != seen, timeout=0.1)
            return self._latest

    def _restart_with_primary(self) -> None:
        # Never tear the session down on the capture thread; grab() runs here.
        try:
            self._capture.stop()
        except Exception:  # noqa: BLE001 - a dead session is fine to replace
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
        self._thread = None
        with self._lock:
            self._latest = None
            self._seq += 1
        self._build_capture(1)

    def close(self) -> None:
        try:
            self._capture.stop()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass


class CaptureManager:
    def __init__(
        self,
        main: CaptureBackend,
        fallback: CaptureBackend | None = None,
        clock=time.monotonic,
    ) -> None:
        self.main = main
        self.fallback = fallback
        self._clock = clock
        self._failures = 0
        self._black = 0
        self._use_fallback = False
        self._health = Health.OK

    @property
    def health(self) -> Health:
        return self._health

    def _activate_fallback(self) -> None:
        if self.fallback is not None:
            self._use_fallback = True
            self._health = Health.WGC_FALLBACK
        else:
            self._health = Health.BLACK

    def grab(self, rect: Rect) -> tuple[np.ndarray | None, Health]:
        if self._use_fallback:
            try:
                frame = self.fallback.grab(rect)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                self._health = Health.BLOCKED
                return None, self._health
            self._health = Health.WGC_FALLBACK
            return frame, self._health

        try:
            frame = self.main.grab(rect)
        except Exception:  # noqa: BLE001 - access lost is expected
            self._failures += 1
            recreate = getattr(self.main, "recreate", None)
            if recreate is not None:
                try:
                    recreate()
                except Exception:  # noqa: BLE001
                    pass
            if self._failures >= 2:
                self._activate_fallback()
                return None, self._health
            self._health = Health.RECREATING
            return None, self._health

        self._failures = 0
        if frame is None:
            self._health = Health.OK
            return None, self._health

        dark = getattr(self.main, "full_dark", None)
        if dark is None:
            luma = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) if frame.ndim == 3 else float(frame.mean())
            dark = luma < 2.0
        if dark:
            self._black += 1
            if self._black >= 3:
                self._activate_fallback()
            else:
                self._health = Health.BLACK
            return None, self._health

        self._black = 0
        self._health = Health.OK
        return frame, self._health

    def close(self) -> None:
        for backend in (self.main, self.fallback):
            if backend is not None:
                try:
                    backend.close()
                except Exception:  # noqa: BLE001
                    pass
