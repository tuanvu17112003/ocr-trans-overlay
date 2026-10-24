"""Rotating file log plus stderr."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class QuietArgos(logging.Filter):
    """Drop argostranslate's per-token INFO.

    That package sets its own logger to INFO at import time, so a level on the
    parent logger cannot silence it; the handler filter can.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.name.startswith("argostranslate")
            and record.levelno < logging.WARNING
        )


def set_debug(enabled: bool) -> None:
    """Verbose logs only while the user asks for them."""
    logging.getLogger().setLevel(logging.DEBUG if enabled else logging.INFO)
    logging.getLogger("argostranslate").setLevel(logging.WARNING)


def setup_logging(level: int = logging.INFO) -> None:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "OCRTrans"
    base.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if os.environ.get("OCRTRANS_DEBUG") else level)
    root.handlers.clear()
    quiet = QuietArgos()

    file_handler = RotatingFileHandler(base / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(quiet)
    root.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    stream.addFilter(quiet)
    root.addHandler(stream)

    # Argos dumps every token and hypothesis at INFO; that floods the log and
    # costs CPU on each translation.
    logging.getLogger("argostranslate").setLevel(logging.WARNING)
