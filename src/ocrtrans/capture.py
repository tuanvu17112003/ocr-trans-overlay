"""Screen capture backends and the DXGI -> WGC watchdog."""

from __future__ import annotations

import threading
import time
from enum import Enum, auto
from typing import Protocol

import cv2
import numpy as np

from .geometry import Rect


class Health(Enum):
    OK = auto()
    RECREATING = auto()
    WGC_FALLBACK = auto()
    BLACK = auto()
    BLOCKED = auto()


class CaptureBackend(Protocol):
    def grab(self, rect: Rect) -> np.ndarray | None: ...

    def close(self) -> None: ...


class DxgiBackend:
    """Desktop Duplication. HDR screens return rgba16f, so always to_bgr()."""

    def __init__(self, monitor_index: int | None = None) -> None:
        from windows_capture import DxgiDuplicationSession

        self._session = DxgiDuplicationSession(monitor_index=monitor_index)

    def grab(self, rect: Rect) -> np.ndarray | None:
        frame = self._session.acquire_frame(timeout_ms=33)
        if frame is None:
            return None
        img = frame.to_bgr()
        return img[rect.y : rect.bottom, rect.x : rect.right]

    def recreate(self) -> None:
        self._session.recreate()

    def close(self) -> None:
        del self._session


class WgcBackend:
    """Windows Graphics Capture of one window or monitor, latest frame kept."""

    def __init__(self, window_name: str | None = None, monitor_index: int | None = None) -> None:
        from windows_capture import WindowsCapture

        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            window_name=window_name,
            monitor_index=monitor_index if window_name is None else None,
        )

        @self._capture.event
        def on_frame_arrived(frame, control) -> None:
            with self._lock:
                self._latest = frame.convert_to_bgr().copy()

        @self._capture.event
        def on_closed() -> None:
            pass

        self._thread = threading.Thread(target=self._capture.start, daemon=True)
        self._thread.start()

    def grab(self, rect: Rect) -> np.ndarray | None:
        with self._lock:
            img = self._latest
        if img is None:
            return None
        return img[rect.y : rect.bottom, rect.x : rect.right]

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

        luma = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) if frame.ndim == 3 else float(frame.mean())
        if luma < 2.0:
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
