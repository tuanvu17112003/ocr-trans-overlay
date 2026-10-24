"""Small click-through performance HUD."""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QLabel, QWidget

from .exclusion import exclude_from_capture


class HudWindow(QWidget):
    def __init__(self, stats_provider: Callable[[], dict]) -> None:
        super().__init__()
        self.stats_provider = stats_provider
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.label = QLabel(self)
        self.label.setStyleSheet(
            "background: rgba(16, 16, 20, 200); color: #E0E0E0; padding: 6px;"
        )
        font = self.label.font()
        font.setFamily("Consolas")
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.move(0, 0)
        self.setGeometry(12, 12, 340, 90)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def set_visible(self, visible: bool) -> None:
        if visible:
            self.show()
            exclude_from_capture(int(self.winId()))
            self.raise_()
        else:
            self.hide()

    def refresh(self) -> None:
        try:
            stats = self.stats_provider()
        except Exception:  # noqa: BLE001 - HUD must never break the app
            return
        p50 = stats.get("p50", "-")
        p95 = stats.get("p95", "-")
        lines = [
            f"engine {stats.get('engine', '-')}   health {stats.get('health', '-')}",
            f"latency p50 {p50} ms   p95 {p95} ms",
            f"ocr {stats.get('ocr_passes', 0)} passes  q{stats.get('ocr_queue', 0)}"
            f"  mt q{stats.get('mt_queue', 0)}  inflight {stats.get('inflight', 0)}",
            f"truncated {stats.get('fit_truncated', 0)}   exclusion {stats.get('exclusion_missed', 0)}",
        ]
        if stats.get("ocr_saturated"):
            lines.append("ocr-saturated")
        if stats.get("mt_saturated"):
            lines.append("mt-saturated")
        self.label.setText("\n".join(lines))
        self.label.adjustSize()
        self.resize(self.label.size())
