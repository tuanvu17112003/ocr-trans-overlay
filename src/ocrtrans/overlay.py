"""Click-through overlay window that paints translated text over the source."""

from __future__ import annotations

import ctypes
import logging
import queue
import time
from dataclasses import dataclass

from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import QWidget

from .config import Config
from .fit import fit_text
from .geometry import Rect
from .lifecycle import OverlayState
from .pipeline import PipelineEvent, RenderItem

log = logging.getLogger(__name__)

WDA_EXCLUDEFROMCAPTURE = 0x00000011
GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080


@dataclass
class _Layout:
    item: RenderItem
    rect: Rect
    font_px: int
    lines: list[str]


class QtMeasure:
    def __init__(self, family: str) -> None:
        self.family = family

    def _metrics(self, font_px: int) -> QFontMetrics:
        return QFontMetrics(QFont(self.family, font_px))

    def measure(self, text: str, font_px: int) -> tuple[int, int]:
        metrics = self._metrics(font_px)
        return metrics.horizontalAdvance(text), metrics.height()

    def line_height(self, font_px: int) -> int:
        return self._metrics(font_px).height()


def _exclude_from_capture(hwnd: int) -> None:
    user32 = ctypes.windll.user32
    user32.GetWindowLongPtrW.restype = ctypes.c_longlong
    user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.SetWindowLongPtrW.restype = ctypes.c_longlong
    user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
    user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)


class OverlayWindow(QWidget):
    def __init__(self, screen: Rect, config: Config, events_queue: queue.Queue) -> None:
        super().__init__()
        self.screen_rect = screen
        self.config = config
        self.events_queue = events_queue
        self.measure = QtMeasure(config.font_family)
        self.state = OverlayState()
        self._items: list[RenderItem] = []
        self._regions: list[Rect] = []
        self._truncated_logged: set[int] = set()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(screen.x, screen.y, screen.w, screen.h)
        self.show()
        _exclude_from_capture(int(self.winId()))

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

    # ---- public API -------------------------------------------------------

    def show_items(self, items: list[RenderItem]) -> None:
        self._items = list(items)
        self.update()

    def clear(self) -> None:
        self._items = []
        self.state.clear()
        self.update()

    def set_regions(self, rects: list[Rect]) -> None:
        self._regions = list(rects)

    def close(self) -> None:
        self._timer.stop()
        super().close()

    # ---- event pump -------------------------------------------------------

    def _drain(self) -> None:
        changed = False
        while True:
            try:
                event: PipelineEvent = self.events_queue.get_nowait()
            except queue.Empty:
                break
            items = self.state.on_event(event, time.monotonic())
            self._items = items
            changed = True
        if changed:
            self.update()

    # ---- painting ---------------------------------------------------------

    def _layout(self) -> list[_Layout]:
        layouts: list[_Layout] = []
        for index, item in enumerate(self._items):
            obstacles = [other.rect for j, other in enumerate(self._items) if j != index]
            fit = fit_text(
                item.text,
                item.rect,
                self.screen_rect,
                self.measure,
                max_growth=self.config.max_growth,
                min_font_px=self.config.min_font_px,
                obstacles=obstacles,
            )
            if fit.truncated and item.block_id not in self._truncated_logged:
                self._truncated_logged.add(item.block_id)
                log.warning("block %s truncated at the font floor", item.block_id)
            layouts.append(_Layout(item=item, rect=fit.rect, font_px=fit.font_px, lines=fit.lines))
        return layouts

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        bg = QColor(self.config.bg_color)
        bg.setAlphaF(self.config.overlay_opacity)
        fg = QColor(self.config.text_color)

        for layout in self._layout():
            x = layout.rect.x - self.screen_rect.x
            y = layout.rect.y - self.screen_rect.y
            painter.fillRect(QRect(x, y, layout.rect.w, layout.rect.h), bg)
            painter.setPen(fg)
            font = QFont(self.config.font_family, layout.font_px)
            painter.setFont(font)
            line_h = self.measure.line_height(layout.font_px)
            for index, line in enumerate(layout.lines):
                painter.drawText(
                    QRect(x, y + index * line_h, layout.rect.w, line_h),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                    line,
                )
        painter.end()
