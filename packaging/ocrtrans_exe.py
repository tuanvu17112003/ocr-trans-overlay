"""Executable entry point for the frozen Windows build.

PyInstaller needs a module-level script; the package itself is started with
`python -m ocrtrans.app`, so this forwards to the same main().

A windowed build has `sys.stdout`/`sys.stderr` set to None. The logging setup
and the `--smoke` summary both write to them, so they are replaced with a
devnull stream before anything else runs: the real log lives in
%LOCALAPPDATA%\OCRTrans\app.log.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback


def _ensure_std_streams() -> None:
    """Windowed builds have no console; give the code a writable sink."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _report_fatal(exc: BaseException) -> None:
    """A frozen build has no traceback window: show the error, then log it."""
    import logging

    logging.getLogger("ocrtrans.frozen").exception("fatal startup error")
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication([])
        QMessageBox.critical(
            None,
            "OCRTrans could not start",
            f"{type(exc).__name__}: {exc}\n\n"
            "The full traceback is in:\n"
            "%LOCALAPPDATA%\\\\OCRTrans\\\\app.log",
        )
        del app
    except Exception:  # noqa: BLE001 - the dialog is best effort
        pass


def _main() -> int:
    _ensure_std_streams()
    from ocrtrans.app import main

    return main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        code = _main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - a crash must be visible
        _report_fatal(exc)
        traceback.print_exc()
        code = 1
    sys.exit(code)