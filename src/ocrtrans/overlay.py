"""Click-through overlay window that paints translated text over the source."""

from __future__ import annotations


import logging
import queue
import time
from dataclasses import dataclass

from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import QWidget

from .config import Config
from .exclusion import exclude_from_capture
from .fit import fit_text
from .geometry import Rect, union_all
from .lifecycle import OverlayState
from .pipeline import PipelineEvent, RenderItem

log = logging.getLogger(__name__)

MARGIN_PX = 8

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


class OverlayWindow(QWidget):
    def __init__(self, screen: Rect, config: Config, events_queue: queue.Queue) -> None:
        super().__init__()
        self._full_screen = screen
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
        self.exclusion_ok = exclude_from_capture(int(self.winId()))
        if not self.exclusion_ok:
            log.error("SetWindowDisplayAffinity failed; the overlay will be captured")

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

    # ---- public API -------------------------------------------------------

    @property
    def GROW_PX(self) -> int:
        """Two floor lines of vertical room: a longer translation grows instead
        of losing its tail. Raise it only if 3-line captions start truncating."""
        return 2 * self.config.min_font_px + 8

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
        if not rects:
            return
        box = union_all(rects)
        full = self._full_screen
        x = max(full.x, box.x - MARGIN_PX)
        y = max(full.y, box.y - MARGIN_PX)
        right = min(full.right, box.right + MARGIN_PX)
        bottom = min(full.bottom, box.bottom + MARGIN_PX + self.GROW_PX)
        rect = Rect(x, y, max(1, right - x), max(1, bottom - y))
        if rect == self.screen_rect:
            return
        # A full-screen layered window forces DWM to composite the whole video
        # on every repaint (and can drop the video overlay plane): keep the
        # window on the text area only. The bottom margin also reserves the
        # room a grown cover needs.
        self.screen_rect = rect
        dpr = self.dpr
        log.info("overlay window %s", rect)
        self.setGeometry(
            round(rect.x / dpr),
            round(rect.y / dpr),
            round(rect.w / dpr),
            round(rect.h / dpr),
        )

    def close(self) -> None:
        self._timer.stop()
        super().close()

    # ---- event pump -------------------------------------------------------

    def _drain(self) -> None:
        now = time.monotonic()
        if self.measure.family != self.config.font_family:
            # A font change with a static screen never reaches show_items.
            self.measure = QtMeasure(self.config.font_family, self.dpr)
            self._recompute()
            self.update()
        latest = self._items
        while True:
            try:
                event: PipelineEvent = self.events_queue.get_nowait()
            except queue.Empty:
                break
            latest = self.state.on_event(event, now)
        ticked = self.state.tick(now)
        if ticked is not None:
            latest = ticked
        if latest != self._items:
            if bool(latest) != bool(self._items):
                log.info("overlay %s", "shown" if latest else "hidden")
            damage = self._damage(self._items, latest)
            self._items = latest
            self._recompute()
            if damage is None:
                self.update()
            else:
                self.update(damage)

    def _damage(self, old_items: list[RenderItem], new_items: list[RenderItem]) -> QRect | None:
        """Repaint only the area the covers touch, not the whole overlay."""
        rects = [
            Rect(item.rect.x, item.rect.y, item.rect.w, item.rect.h + self.GROW_PX)
            for item in old_items + new_items
        ]
        if not rects:
            return None
        box = union_all(rects)
        dpr = self.dpr
        return QRect(
            round((box.x - self.screen_rect.x) / dpr) - 2,
            round((box.y - self.screen_rect.y) / dpr) - 2,
            round(box.w / dpr) + 4,
            round(box.h / dpr) + 4,
        )

    # ---- layout -----------------------------------------------------------

    def _fit(self, item: RenderItem, limit_bottom: int) -> _Layout:
        # two rects: cover = the whole OCR block, rect = the text inside it
        inner = Rect(item.rect.x + 2, item.rect.y, max(1, item.rect.w - 4), item.rect.h)
        fit = fit_text(
            item.text,
            inner,
            self.screen_rect,
            self.measure,
            line_px=item.line_px,
            min_font_px=self.config.min_font_px,
            limit_bottom=limit_bottom,
        )
        if fit.truncated and item.block_id not in self._truncated_logged:
            self._truncated_logged.add(item.block_id)
            log.warning("block %s truncated at the font floor", item.block_id)
        y = inner.y + max(0, (inner.h - fit.rect.h) // 2)
        return _Layout(
            cover=Rect(item.rect.x, item.rect.y, item.rect.w, max(item.rect.h, fit.rect.h)),
            rect=Rect(inner.x, y, fit.rect.w, fit.rect.h),
            font_px=fit.font_px,
            lines=fit.lines,
        )

    def _recompute(self) -> None:
        if self.measure.family != self.config.font_family:
            self.measure = QtMeasure(self.config.font_family, self.dpr)
        limits: list[int] = []
        for item in self._items:
            limit = min(self.screen_rect.bottom, item.rect.bottom + self.GROW_PX)
            for other in self._items:
                if other is item:
                    continue
                overlaps_x = min(item.rect.right, other.rect.right) > max(item.rect.x, other.rect.x)
                if overlaps_x and item.rect.y < other.rect.y <= limit:
                    # A cover growing into another block would flicker with it.
                    limit = min(limit, other.rect.y - 4)
            limits.append(limit)
        self._layouts = [self._fit(item, limit) for item, limit in zip(self._items, limits)]

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


