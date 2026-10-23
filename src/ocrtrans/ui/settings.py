"""Settings dialog."""

from __future__ import annotations

from typing import Callable

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from ..config import ENGINES, Config, save_config


class SettingsDialog(QDialog):
    def __init__(
        self,
        config: Config,
        on_save: Callable[[Config], None],
        benchmarks: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.on_save = on_save
        self._benchmarks = benchmarks or {}
        self.setWindowTitle("OCR Translation settings")

        self.engine = QComboBox()
        self.engine.addItems(sorted(ENGINES))
        self.engine.setCurrentText(config.engine)
        self.profile = QComboBox()
        self.profile.addItems(["safe", "fast"])
        self.profile.setCurrentText(config.stability_profile)
        self.fps = QSpinBox()
        self.fps.setRange(5, 30)
        self.fps.setValue(config.capture_fps)
        self.mode = QComboBox()
        self.mode.addItems(["single", "multi", "monitor"])
        self.mode.setCurrentText(config.region_mode)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.6, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(config.overlay_opacity)
        self.min_font = QSpinBox()
        self.min_font.setRange(8, 32)
        self.min_font.setValue(config.min_font_px)
        self.gemini_key = QLineEdit(config.keys.get("gemini", ""))
        self.deepl_key = QLineEdit(config.keys.get("deepl", ""))
        self.consent = QCheckBox("I understand the selected online engine sends text to a third party")
        self.consent.setChecked(config.online_consent)
        self.benchmark = QLabel(self._benchmarks.get(config.engine, "no benchmark yet"))

        form = QFormLayout()
        form.addRow("Engine", self.engine)
        form.addRow("Benchmark", self.benchmark)
        form.addRow("Stability", self.profile)
        form.addRow("Capture fps", self.fps)
        form.addRow("Region mode", self.mode)
        form.addRow("Overlay opacity", self.opacity)
        form.addRow("Minimum font px", self.min_font)
        form.addRow("Gemini key", self.gemini_key)
        form.addRow("DeepL key", self.deepl_key)
        form.addRow(self.consent)

        warning = QLabel(
            "Privacy: Google, MyMemory, Gemini and DeepL receive the text the overlay reads. "
            "Local engines never leave this machine."
        )
        warning.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(warning)
        layout.addWidget(buttons)

        self.engine.currentTextChanged.connect(
            lambda name: self.benchmark.setText(self._benchmarks.get(name, "no benchmark yet"))
        )

    def _save(self) -> None:
        cfg = self.config
        cfg.engine = self.engine.currentText()
        cfg.stability_profile = self.profile.currentText()
        cfg.capture_fps = self.fps.value()
        cfg.region_mode = self.mode.currentText()
        cfg.overlay_opacity = self.opacity.value()
        cfg.min_font_px = self.min_font.value()
        keys = dict(cfg.keys)
        if self.gemini_key.text().strip():
            keys["gemini"] = self.gemini_key.text().strip()
        if self.deepl_key.text().strip():
            keys["deepl"] = self.deepl_key.text().strip()
        cfg.keys = keys
        cfg.online_consent = self.consent.isChecked()
        save_config(cfg)
        self.on_save(cfg)
        self.accept()
