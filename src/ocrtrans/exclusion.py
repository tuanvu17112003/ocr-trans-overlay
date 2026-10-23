"""Capture-exclusion gates for the overlay window."""

from __future__ import annotations

from typing import Protocol


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
