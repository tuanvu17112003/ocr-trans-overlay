"""Full-screen drag selection and handle editing for capture regions."""

from __future__ import annotations

from PyQt6.QtCore import QEventLoop, QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..geometry import Rect
from ..regions import drag_rect, handle_at

HANDLE_PX = 12
ACCENT = QColor(94, 224, 176)


class RegionPicker(QWidget):
    def __init__(self, screen: Rect) -> None:
        super().__init__()
        self.screen_rect = screen
        self._start: QPoint | None = None
        self._end: QPoint | None = None
        self._result: list[Rect] = []
        self._loop = QEventLoop()
        self._mode = "pick"
        self._target: Rect | None = None
        self._adjusted: Rect | None = None
        self._drag_handle: str | None = None
        self._drag_base: Rect | None = None
        self._drag_origin: tuple[int, int] | None = None
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(screen.x, screen.y, screen.w, screen.h)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def pick(self) -> list[Rect]:
        self.show()
        self._loop.exec()
        return self._result

    def adjust(self, rect: Rect) -> Rect | None:
        """Edit one rect with 8 handles; Enter accepts, Esc keeps the old one."""
        self._mode = "adjust"
        self._target = rect
        self._adjusted = None
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show()
        self.activateWindow()
        self.setFocus()
        self.grabKeyboard()
        self._loop.exec()
        return self._adjusted

    def _screen_point(self, event) -> tuple[int, int]:
        point = event.position().toPoint()
        return self.screen_rect.x + point.x(), self.screen_rect.y + point.y()

    def _local(self, rect: Rect) -> QRect:
        return QRect(rect.x - self.screen_rect.x, rect.y - self.screen_rect.y, rect.w, rect.h)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._mode == "adjust":
            if self._target is None:
                return
            point = self._screen_point(event)
            handle = handle_at(self._target, point)
            if handle is None:
                return
            self._drag_handle = handle
            self._drag_base = self._target
            self._drag_origin = point
            return
        self._start = event.position().toPoint()
        self._end = self._start
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._mode == "adjust":
            if self._drag_handle is None or self._drag_base is None or self._drag_origin is None:
                return
            x, y = self._screen_point(event)
            dx = x - self._drag_origin[0]
            dy = y - self._drag_origin[1]
            self._target = drag_rect(self._drag_base, self._drag_handle, dx, dy, self.screen_rect)
            self.update()
            return
        if self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mode == "adjust":
            self._drag_handle = None
            return
        if self._start is None:
            return
        self._end = event.position().toPoint()
        rect = QRect(self._start, self._end).normalized()
        if rect.width() > 8 and rect.height() > 8:
            self._result = [
                Rect(
                    self.screen_rect.x + rect.x(),
                    self.screen_rect.y + rect.y(),
                    rect.width(),
                    rect.height(),
                )
            ]
        self.close()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._mode == "adjust":
            self._accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._mode != "adjust":
            super().keyPressEvent(event)
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def _accept(self) -> None:
        self._adjusted = self._target
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._mode == "adjust":
            self.releaseKeyboard()
        self._loop.quit()
        super().closeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
        if self._mode == "adjust" and self._target is not None:
            self._paint_target(painter)
            painter.end()
            return
        if self._start is not None and self._end is not None:
            selection = QRect(self._start, self._end).normalized()
            painter.fillRect(selection, QColor(255, 255, 255, 30))
            painter.setPen(QPen(ACCENT, 2))
            painter.drawRect(selection)
        painter.end()

    def _paint_target(self, painter: QPainter) -> None:
        rect = self._local(self._target)
        painter.fillRect(rect, QColor(94, 224, 176, 30))
        painter.setPen(QPen(ACCENT, 2))
        painter.drawRect(rect)
        half = HANDLE_PX // 2
        points = [
            (rect.left(), rect.top()),
            (rect.center().x(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.center().y()),
            (rect.right(), rect.bottom()),
            (rect.center().x(), rect.bottom()),
            (rect.left(), rect.bottom()),
            (rect.left(), rect.center().y()),
        ]
        painter.setBrush(ACCENT)
        for x, y in points:
            painter.drawRect(x - half, y - half, HANDLE_PX, HANDLE_PX)


def pick_regions(screen: Rect) -> list[Rect]:
    return RegionPicker(screen).pick()


def adjust_rect(screen: Rect, rect: Rect) -> Rect | None:
    return RegionPicker(screen).adjust(rect)
