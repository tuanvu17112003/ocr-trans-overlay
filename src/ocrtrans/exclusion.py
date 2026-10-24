"""Capture-exclusion gates for the overlay window."""

from __future__ import annotations

import ctypes
from typing import Protocol

WDA_EXCLUDEFROMCAPTURE = 0x00000011
GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080


def exclude_from_capture(hwnd: int) -> bool:
    """Make the window click-through and invisible to screen capture APIs."""
    user32 = ctypes.windll.user32
    user32.GetWindowLongPtrW.restype = ctypes.c_longlong
    user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.SetWindowLongPtrW.restype = ctypes.c_longlong
    user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]
    user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.SetWindowDisplayAffinity.restype = ctypes.c_int
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


class HideBridge(Protocol):
    def hide_and_ack(self, timeout_ms: int) -> bool: ...

    def restore(self) -> None: ...


class AffinityGate:
    """Primary path: WDA_EXCLUDEFROMCAPTURE already hides the window."""

    def acquire(self, deadline_ms: int = 8) -> bool:
        return True

    def release(self) -> None:
        pass


class QtHideGate:
    """Fallback: hide the window, wait for the compositor, then capture."""

    def __init__(self, bridge: HideBridge, deadline_ms: int = 8) -> None:
        self.bridge = bridge
        self.deadline_ms = deadline_ms
        self.tripped = False
        self._failures = 0

    def acquire(self, deadline_ms: int | None = None) -> bool:
        ok = self.bridge.hide_and_ack(deadline_ms or self.deadline_ms)
        if ok:
            self._failures = 0
        else:
            self._failures += 1
            if self._failures >= 3:
                self.tripped = True
        return ok

    def release(self) -> None:
        self.bridge.restore()
