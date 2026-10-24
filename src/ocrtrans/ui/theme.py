"""Dark theme tokens and the single application stylesheet."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

BG = "#12141A"
PANEL = "#171A22"
BORDER = "#242833"
TEXT = "#E7EAF0"
MUTED = "#8B93A5"
ACCENT = "#5EE0B0"
DANGER = "#FF8F8F"
SELECTED = "#1B2430"

QSS = f"""
QWidget#ControlWindow {{
    background: {BG};
    color: {TEXT};
    font-family: 'Segoe UI';
    font-size: 12px;
}}
QWidget#ControlWindow QLabel {{ color: {TEXT}; }}
QWidget#ControlWindow QLabel#title {{ font-size: 15px; font-weight: 600; }}
QWidget#ControlWindow QLabel#muted {{ color: {MUTED}; }}
QWidget#ControlWindow QLabel#danger {{ color: {DANGER}; }}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    color: {MUTED};
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}
QListWidget {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
}}
QListWidget::item {{ padding: 6px 8px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {SELECTED}; color: {ACCENT}; }}
QPushButton {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 10px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {BORDER}; }}
QPushButton#primary {{
    background: {ACCENT};
    color: #08120E;
    font-weight: 600;
    border: none;
    padding: 9px 12px;
}}
QPushButton#primary:disabled {{ background: {PANEL}; color: {MUTED}; }}
QPushButton#danger {{ color: {DANGER}; }}
QPushButton#danger:disabled {{ color: {MUTED}; }}
QComboBox, QLineEdit, QSpinBox, QFontComboBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QCheckBox {{ color: {TEXT}; }}
QCheckBox:disabled {{ color: {MUTED}; }}
"""


def apply(app: QApplication) -> None:
    app.setStyleSheet(QSS)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(PANEL))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(PANEL))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#08120E"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    app.setPalette(palette)
