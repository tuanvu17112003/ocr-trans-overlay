"""On-screen outlines for the capture regions (shown while stopped)."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..exclusion import exclude_from_capture
from ..geometry import Rect, union_all

log = logging.getLogger(__name__)

ACCENT = QColor("#5EE0B0")
MUTED = QColor("#8B93A5")
BADGE_TEXT = QColor("#08120E")
MARGIN_PX = 12
BADGE_W = 20
BADGE_H = 16


class RegionOutline(QWidget):
    """Frameless click-through window that frames every capture region.

    shortcut: screen coordinates are logical pixels and the target machine runs
    at dpr 1.0 (out of scope in the spec); divide by dpr here if HiDPI comes back.
    """

    def __init__(self, screen: Rect) -> None:
        super().__init__()
        self.screen_rect = screen
        self.rects: list[Rect] = []
        self.active: int | None = None
        self.inactive: set[int] = set()
        self.excluded = False
        self._origin = (screen.x, screen.y)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._fit_geometry()
        self.hide()

    def set_regions(self, rects: list[Rect], active: int | None, inactive: set[int]) -> None:
        self.rects = list(rects)
        self.active = active
        self.inactive = set(inactive)
        self._fit_geometry()
        self.update()

    def set_visible_for(self, stopped: bool, force: bool) -> bool:
        if not (stopped or force) or not self.rects:
            self.hide()
            return False
        if not self.excluded:
            self.excluded = bool(exclude_from_capture(int(self.winId())))
            if not self.excluded:
                log.error("the region outline cannot be excluded from capture")
        if not self.excluded:
            self.hide()
            return False
        self.show()
        return self.isVisible()

    def _fit_geometry(self) -> None:
        # Only cover the regions: a full-screen layered window makes DWM
        # recomposite the video output (the overlay suffered from this).
        box = union_all(self.rects) if self.rects else self.screen_rect
        full = self.screen_rect
        x = max(full.x, box.x - MARGIN_PX)
        y = max(full.y, box.y - BADGE_H - MARGIN_PX)
        right = min(full.right, box.right + MARGIN_PX)
        bottom = min(full.bottom, box.bottom + MARGIN_PX)
        self._origin = (x, y)
        self.setGeometry(x, y, max(1, right - x), max(1, bottom - y))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        if not self.rects:
            return
        painter = QPainter(self)
        ox, oy = self._origin
        for index, rect in enumerate(self.rects):
            active = index == self.active
            pen = QPen(ACCENT if active else MUTED, 3 if active else 1)
            if index in self.inactive:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            x, y = rect.x - ox, rect.y - oy
            painter.drawRect(x, y, rect.w, rect.h)
            self._draw_badge(painter, x, y, index + 1, active)

    def _draw_badge(self, painter: QPainter, x: int, y: int, number: int, active: bool) -> None:
        bx, by = x - 2, y - BADGE_H - 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ACCENT if active else MUTED)
        painter.drawRoundedRect(bx, by, BADGE_W, BADGE_H, 4, 4)
        font = QFont()
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(BADGE_TEXT)
        painter.drawText(QRect(bx, by, BADGE_W, BADGE_H), Qt.AlignmentFlag.AlignCenter, str(number))
