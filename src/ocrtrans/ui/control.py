"""Main control window: every user-facing setting, Start/Stop and status."""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import ENGINES, Config
from ..geometry import Rect
from ..regions import NormalizedRegion, to_physical
from . import theme

ONLINE_ENGINES = {"google", "gemini", "deepl", "mymemory"}


class ControlWindow(QWidget):
    def __init__(
        self,
        config: Config,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_add_region: Callable[[], None],
        on_adjust_region: Callable[[int], None],
        on_delete_region: Callable[[int], None],
        on_apply: Callable[[Config], None],
        status_provider: Callable[[], dict],
        monitor: Rect,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_add_region = on_add_region
        self.on_adjust_region = on_adjust_region
        self.on_delete_region = on_delete_region
        self.on_apply = on_apply
        self.status_provider = status_provider
        self.on_close = on_close
        self.monitor = monitor
        self.running = False

        self.setWindowTitle("OCRTrans - screen translation")
        self.setObjectName("ControlWindow")
        self.resize(520, 0)

        self.engine = QComboBox()
        self.engine.addItems(sorted(ENGINES))
        self.engine.setCurrentText(config.engine)
        self.gemini_key = QLineEdit(config.keys.get("gemini", ""))
        self.deepl_key = QLineEdit(config.keys.get("deepl", ""))
        self.deepl_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.consent = QCheckBox("Allow captured text to reach an online provider")
        self.consent.setChecked(config.online_consent)
        self.profile = QComboBox()
        self.profile.addItems(["safe", "fast"])
        self.profile.setCurrentText(config.stability_profile)
        self.fps = QSpinBox()
        self.fps.setRange(5, 30)
        self.fps.setValue(config.capture_fps)
        self.mode = QComboBox()
        self.mode.addItems(["single", "multi"])
        self.mode.setCurrentText(config.region_mode)
        self.min_font = QSpinBox()
        self.min_font.setRange(8, 32)
        self.min_font.setValue(config.min_font_px)

        self.region_list = QListWidget()
        self.add_button = QPushButton("Add region")
        self.adjust_button = QPushButton("Adjust")
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("danger")
        self.add_button.clicked.connect(lambda: self.on_add_region())
        self.adjust_button.clicked.connect(self._adjust_region)
        self.delete_button.clicked.connect(self._delete_region)
        self.region_list.currentRowChanged.connect(lambda _row: self._refresh_region_buttons())
        self.mode.currentTextChanged.connect(lambda _mode: self.refresh_regions())
        self.show_outlines = QCheckBox("Show outlines while running")

        self._text_color = config.text_color
        self._bg_color = config.bg_color
        self.text_color_button = QPushButton()
        self.bg_color_button = QPushButton()
        self.text_color_button.clicked.connect(self._pick_text_color)
        self.bg_color_button.clicked.connect(self._pick_bg_color)
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(config.font_family))
        self.preview = QLabel("Đây là bản dịch mẫu")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(56)

        self.chip = QLabel("? stopped")
        self.chip.setObjectName("muted")
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(36)
        self.start_button.clicked.connect(self._toggle)
        self.status_label = QLabel("stopped")
        self.status_label.setObjectName("muted")
        self.status_label.setWordWrap(True)

        title = QLabel("OCRTrans")
        title.setObjectName("title")
        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.chip)

        regions_box = QGroupBox("REGIONS")
        regions_layout = QVBoxLayout(regions_box)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        mode_row.addWidget(self.mode)
        mode_row.addStretch()
        mode_row.addWidget(self.show_outlines)
        regions_layout.addLayout(mode_row)
        self.region_hint = QLabel("No region yet — Add one")
        self.region_hint.setObjectName("muted")
        regions_layout.addWidget(self.region_hint)
        regions_layout.addWidget(self.region_list)
        button_row = QHBoxLayout()
        button_row.addWidget(self.add_button)
        button_row.addWidget(self.adjust_button)
        button_row.addWidget(self.delete_button)
        regions_layout.addLayout(button_row)

        overlay_box = QGroupBox("OVERLAY")
        overlay_layout = QVBoxLayout(overlay_box)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Text"))
        color_row.addWidget(self.text_color_button)
        color_row.addWidget(QLabel("Background"))
        color_row.addWidget(self.bg_color_button)
        color_row.addStretch()
        overlay_layout.addLayout(color_row)
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font"))
        font_row.addWidget(self.font_combo, 1)
        font_row.addWidget(QLabel("Min px"))
        font_row.addWidget(self.min_font)
        overlay_layout.addLayout(font_row)
        overlay_layout.addWidget(self.preview)

        engine_box = QGroupBox("ENGINE")
        engine_form = QFormLayout(engine_box)
        engine_form.addRow("Engine", self.engine)
        engine_form.addRow("Gemini key", self.gemini_key)
        engine_form.addRow("DeepL key", self.deepl_key)
        engine_form.addRow("", self.consent)
        engine_form.addRow("Stability", self.profile)
        engine_form.addRow("FPS", self.fps)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(regions_box)
        layout.addWidget(overlay_box)
        layout.addWidget(engine_box)
        layout.addWidget(self.start_button)
        layout.addWidget(self.status_label)

        self.set_appearance(
            config.text_color, config.bg_color, config.font_family, config.min_font_px
        )
        self.min_font.valueChanged.connect(lambda _value: self._apply_appearance())
        self.font_combo.currentFontChanged.connect(self._on_font_changed)
        self.refresh_regions()

        self.engine.currentTextChanged.connect(self._sync_keys)
        self._sync_keys(self.engine.currentText())

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _sync_keys(self, engine: str) -> None:
        online = engine in ONLINE_ENGINES
        self.gemini_key.setEnabled(engine == "gemini")
        self.deepl_key.setEnabled(engine == "deepl")
        self.consent.setEnabled(online)

    def _toggle(self) -> None:
        if self.running:
            self.set_running(False)
            self.on_stop()
        else:
            self.set_running(True)
            self.on_start()

    def set_running(self, running: bool) -> None:
        self.running = running
        self.start_button.setText("Stop" if running else "Start")
        self.chip.setText("? running" if running else "? stopped")
        self.chip.setStyleSheet(f"color: {theme.ACCENT if running else theme.MUTED}")

    def set_appearance(
        self, text_color: str, bg_color: str, font_family: str, min_font_px: int
    ) -> None:
        self._text_color = text_color
        self._bg_color = bg_color
        self.font_combo.setCurrentFont(QFont(font_family))
        # The combo falls back to the platform default when the font database is
        # empty (offscreen tests), so keep the requested family as the value.
        self._font_family = font_family
        self.min_font.setValue(min_font_px)
        self._sync_swatches()
        self._paint_preview()

    def _on_font_changed(self, font: QFont) -> None:
        self._font_family = font.family()
        self._apply_appearance()

    def _apply_appearance(self) -> None:
        self._paint_preview()
        self.on_apply(self.read_config())

    def _sync_swatches(self) -> None:
        for button, color, label in (
            (self.text_color_button, self._text_color, "Text"),
            (self.bg_color_button, self._bg_color, "Background"),
        ):
            ink = "#08120E" if QColor(color).lightness() > 128 else theme.TEXT
            button.setText(f"{label} {color}")
            button.setStyleSheet(
                f"background: {color}; color: {ink}; border: 1px solid {theme.BORDER};"
                " border-radius: 6px; padding: 4px 8px;"
            )

    def _paint_preview(self) -> None:
        family = self._font_family
        px = max(12, self.min_font.value())
        self.preview.setStyleSheet(
            f"background: {self._bg_color}; color: {self._text_color};"
            f" font-family: '{family}'; font-size: {px}px;"
            f" border: 1px solid {theme.BORDER}; border-radius: 8px; padding: 10px;"
        )

    def _pick_text_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._text_color), self, "Overlay text color")
        if color.isValid():
            self.set_appearance(
                color.name(),
                self._bg_color,
                self._font_family,
                self.min_font.value(),
            )
            self.on_apply(self.read_config())

    def _pick_bg_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._bg_color), self, "Overlay background color")
        if color.isValid():
            self.set_appearance(
                self._text_color,
                color.name(),
                self._font_family,
                self.min_font.value(),
            )
            self.on_apply(self.read_config())

    def selected_region(self) -> int | None:
        row = self.region_list.currentRow()
        return row if row >= 0 else None

    def refresh_regions(self) -> None:
        current = self.selected_region()
        self.region_list.clear()
        self.region_hint.setVisible(not self.config.regions)
        applied = 1 if self.mode.currentText() == "single" else len(self.config.regions)
        for index, raw in enumerate(self.config.regions):
            rect = to_physical(NormalizedRegion(**raw), self.monitor)
            text = f"{index + 1}  ·  {rect.w} × {rect.h}"
            if index >= applied:
                text += "  (inactive)"
            self.region_list.addItem(text)
        if current is not None and self.region_list.count():
            self.region_list.setCurrentRow(min(current, self.region_list.count() - 1))
        self._refresh_region_buttons()

    def _refresh_region_buttons(self) -> None:
        selected = self.selected_region() is not None
        self.adjust_button.setEnabled(selected)
        self.delete_button.setEnabled(selected and len(self.config.regions) > 1)
        self.start_button.setEnabled(bool(self.config.regions))

    def _adjust_region(self) -> None:
        index = self.selected_region()
        if index is not None:
            self.on_adjust_region(index)

    def _delete_region(self) -> None:
        index = self.selected_region()
        if index is not None:
            self.on_delete_region(index)

    def set_status_text(self, text: str) -> None:
        self.status_label.setText(text)

    def _refresh(self) -> None:
        try:
            stats = self.status_provider()
        except Exception:  # noqa: BLE001 - the window must never break the app
            return
        self.set_status(stats)

    def set_status(self, stats: dict) -> None:
        ocr_errors = stats.get("ocr_errors", 0)
        mt_errors = stats.get("mt_errors", 0)
        error_text = f"   err {ocr_errors}/{mt_errors}" if (ocr_errors or mt_errors) else ""
        text = (
            f"engine {stats.get('engine', self.engine.currentText())}   "
            f"{stats.get('health', '-')}\n"
            f"OCR {stats.get('ocr_passes', 0)} pass "
            f"({stats.get('scan_rate', 0):.1f}/s)   "
            f"text {stats.get('blocks', 0)}{error_text}   "
            f"shown {stats.get('rendered', 0)}   tiled {stats.get('tiled', 0)}\n"
            f"queue {stats.get('ocr_queue', 0)}   MT {stats.get('mt_queue', 0)}   "
            f"scan {stats.get('last_scan_ms', 0):.0f} ms\n"
            f"p50 {stats.get('p50', '-')} ms   p95 {stats.get('p95', '-')} ms   "
            f"hidden {stats.get('hidden_stale', 0)}"
        )
        if stats.get("error"):
            text += f"\n{stats['error']}"
        self.status_label.setText(text)

    def read_config(self) -> Config:
        cfg = self.config
        cfg.engine = self.engine.currentText()
        cfg.stability_profile = self.profile.currentText()
        cfg.capture_fps = self.fps.value()
        cfg.region_mode = self.mode.currentText()
        cfg.min_font_px = self.min_font.value()
        cfg.text_color = self._text_color
        cfg.bg_color = self._bg_color
        cfg.font_family = self._font_family
        keys = dict(cfg.keys)
        if self.gemini_key.text().strip():
            keys["gemini"] = self.gemini_key.text().strip()
        if self.deepl_key.text().strip():
            keys["deepl"] = self.deepl_key.text().strip()
        cfg.keys = keys
        cfg.online_consent = self.consent.isChecked()
        return cfg

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.on_apply(self.read_config())
        if self.on_close is not None:
            self.on_close()
        super().closeEvent(event)


