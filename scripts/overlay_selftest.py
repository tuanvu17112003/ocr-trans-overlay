"""Show a fixed Vietnamese paragraph over a dummy window and verify exclusion."""

from __future__ import annotations

import argparse
import ctypes
import os
import queue
import sys
import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QWidget

from ocrtrans.capture import DxgiBackend
from ocrtrans.config import Config
from ocrtrans.geometry import Rect
from ocrtrans.overlay import WDA_EXCLUDEFROMCAPTURE, OverlayWindow
from ocrtrans.pipeline import RenderItem

FULL = Rect(0, 0, 10_000, 10_000)
SCREEN = Rect(0, 0, 2560, 1440)
TEXT = "Xin chào, lữ khách. Cổng đang mở."


def capture_magenta(backend: DxgiBackend) -> float:
    frame = None
    for _ in range(20):
        candidate = backend.grab(FULL)
        if candidate is not None and float(candidate.mean()) > 2.0:
            frame = candidate
            break
        time.sleep(0.05)
    if frame is None:
        return -1.0
    mask = (frame[:, :, 0] > 200) & (frame[:, :, 2] > 200) & (frame[:, :, 1] < 60)
    return float(mask.mean())


def settle(app: QApplication, seconds: float = 0.5) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-capture", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="auto-quit after N seconds")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    config = Config()
    config.bg_color = "#FF00FF"
    config.overlay_opacity = 1.0

    dummy = QWidget()
    dummy.setStyleSheet("background: #000000;")
    dummy.setGeometry(200, 200, 600, 300)
    dummy.show()

    overlay = OverlayWindow(SCREEN, config, queue.Queue())
    overlay.show_items(
        [RenderItem(1, TEXT, Rect(240, 260, 520, 120), "Hello, traveler. The gate is open.", False)]
    )
    settle(app)

    if not args.verify_capture:
        if args.duration > 0:
            QTimer.singleShot(int(args.duration * 1000), app.quit)
        return app.exec()

    backend = DxgiBackend()
    hwnd = int(overlay.winId())
    ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0)
    settle(app)
    control = capture_magenta(backend)

    ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    settle(app)
    after = capture_magenta(backend)

    gate_ratio = None
    if os.environ.get("OCRTRANS_FORCE_HIDE_GATE") == "1":
        overlay.hide()
        settle(app)
        gate_ratio = capture_magenta(backend)

    print(f"control={control:.4%} after_affinity={after:.4%} hide_gate={gate_ratio}")
    ok = control > 0.001 and after < 0.0002 and (gate_ratio is None or gate_ratio < 0.0002)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
