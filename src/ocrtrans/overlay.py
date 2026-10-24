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
    cover: Rect
    rect: Rect
    font_px: int
    lines: list[str]


class QtMeasure:
    """Measure in physical pixels; dpr maps Qt's logical pixels to physical."""

    def __init__(self, family: str, dpr: float = 1.0) -> None:
        self.family = family
        self.dpr = dpr
        self._cache: dict[int, QFontMetrics] = {}

    def _metrics(self, font_px: int) -> QFontMetrics:
        key = max(1, round(font_px))
        if key not in self._cache:
            font = QFont(self.family)
            font.setPixelSize(key)
            self._cache[key] = QFontMetrics(font)
        return self._cache[key]

    def measure(self, text: str, font_px: int) -> tuple[int, int]:
        metrics = self._metrics(font_px)
        return (
            round(metrics.horizontalAdvance(text) * self.dpr),
            round(metrics.height() * self.dpr),
        )

    def line_height(self, font_px: int) -> int:
        return round(self._metrics(font_px).height() * self.dpr)


def _exclude_from_capture(hwnd: int) -> bool:
    user32 = ctypes.windll.user32
    user32.GetWindowLongPtrW.restype = ctypes.c_longlong
    user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.SetWindowLongPtrW.restype = ctypes.c_longlong
    user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


class OverlayWindow(QWidget):
    def __init__(self, screen: Rect, config: Config, events_queue: queue.Queue) -> None:
        super().__init__()
        self.screen_rect = screen
        self.config = config
        self.events_queue = events_queue
        self.dpr = max(1.0, float(self.devicePixelRatioF()))
        self.measure = QtMeasure(config.font_family, self.dpr)
        self.state = OverlayState()
        self._items: list[RenderItem] = []
        self._layouts: list[_Layout] = []
        self._regions: list[Rect] = []
        self._truncated_logged: set[int] = set()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(
            round(screen.x / self.dpr),
            round(screen.y / self.dpr),
            round(screen.w / self.dpr),
            round(screen.h / self.dpr),
        )
        self.show()
        self.exclusion_ok = _exclude_from_capture(int(self.winId()))
        if not self.exclusion_ok:
            log.error("SetWindowDisplayAffinity failed; the overlay will be captured")

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

    # ---- public API -------------------------------------------------------

    def show_items(self, items: list[RenderItem]) -> None:
        self._items = list(items)
        self._recompute()
        self.update()

    @property
    def shown(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items = []
        self._layouts = []
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
        now = time.monotonic()
        while True:
            try:
                event: PipelineEvent = self.events_queue.get_nowait()
            except queue.Empty:
                break
            self._items = self.state.on_event(event, now)
            changed = True
        ticked = self.state.tick(now)
        if ticked is not None:
            self._items = ticked
            changed = True
        if changed:
            self._recompute()
            self.update()

    # ---- layout -----------------------------------------------------------

    def _fit(self, item: RenderItem) -> _Layout:
        # two rects: cover = the whole OCR block, rect = the text inside it
        inner = Rect(item.rect.x + 2, item.rect.y, max(1, item.rect.w - 4), item.rect.h)
        fit = fit_text(
            item.text,
            inner,
            self.screen_rect,
            self.measure,
            line_px=item.line_px,
            min_font_px=self.config.min_font_px,
        )
        if fit.truncated and item.block_id not in self._truncated_logged:
            self._truncated_logged.add(item.block_id)
            log.warning("block %s truncated at the font floor", item.block_id)
        y = inner.y + max(0, (inner.h - fit.rect.h) // 2)
        return _Layout(
            cover=item.rect,
            rect=Rect(inner.x, y, fit.rect.w, fit.rect.h),
            font_px=fit.font_px,
            lines=fit.lines,
        )

    def _recompute(self) -> None:
        self._layouts = [self._fit(item) for item in self._items]

    # ---- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        bg = QColor(self.config.bg_color)
        fg = QColor(self.config.text_color)
        dpr = self.dpr

        for layout in self._layouts:
            cover = QRect(
                round((layout.cover.x - self.screen_rect.x) / dpr),
                round((layout.cover.y - self.screen_rect.y) / dpr),
                round(layout.cover.w / dpr),
                round(layout.cover.h / dpr),
            )
            painter.fillRect(cover, bg)
            painter.save()
            painter.setClipRect(cover)
            x = round((layout.rect.x - self.screen_rect.x) / dpr)
            y = round((layout.rect.y - self.screen_rect.y) / dpr)
            w = round(layout.rect.w / dpr)
            painter.setPen(fg)
            font = QFont(self.config.font_family)
            font.setPixelSize(max(1, round(layout.font_px / dpr)))
            painter.setFont(font)
            line_h = max(1, round(self.measure.line_height(layout.font_px) / dpr))
            for index, line in enumerate(layout.lines):
                painter.drawText(
                    QRect(x, y + index * line_h, w, line_h),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                    line,
                )
            painter.restore()
        painter.end()


