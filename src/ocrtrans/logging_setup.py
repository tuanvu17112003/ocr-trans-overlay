"""Rotating file log plus stderr."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "OCRTrans"
    base.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    file_handler = RotatingFileHandler(base / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    # Argos dumps every token and hypothesis at INFO; that floods the log and
    # costs CPU on each translation.
    logging.getLogger("argostranslate").setLevel(logging.WARNING)
