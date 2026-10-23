"""Full-screen drag selection for capture regions."""

from __future__ import annotations

from PyQt6.QtCore import QEventLoop, QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..geometry import Rect


class RegionPicker(QWidget):
    def __init__(self, screen: Rect) -> None:
        super().__init__()
        self.screen_rect = screen
        self._start: QPoint | None = None
        self._end: QPoint | None = None
        self._result: list[Rect] = []
        self._loop = QEventLoop()
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

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._start = event.position().toPoint()
        self._end = self._start
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
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

    def closeEvent(self, event) -> None:  # noqa: N802
        self._loop.quit()
        super().closeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
        if self._start is not None and self._end is not None:
            selection = QRect(self._start, self._end).normalized()
            painter.fillRect(selection, QColor(255, 255, 255, 30))
            painter.setPen(QPen(QColor(0, 200, 255), 2))
            painter.drawRect(selection)
        painter.end()


def pick_regions(screen: Rect) -> list[Rect]:
    return RegionPicker(screen).pick()
