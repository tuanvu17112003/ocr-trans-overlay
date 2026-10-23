"""Global hotkeys via RegisterHotKey (no keyboard hooks)."""

from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes
from typing import Callable

log = logging.getLogger(__name__)

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

_MODS = {
    "CTRL": MOD_CONTROL,
    "CONTROL": MOD_CONTROL,
    "ALT": MOD_ALT,
    "SHIFT": MOD_SHIFT,
    "WIN": MOD_WIN,
}


def parse_binding(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in text.split("+") if part.strip()]
    if not parts:
        raise ValueError("empty hotkey binding")
    mods = 0
    key: int | None = None
    for part in parts:
        upper = part.upper()
        if upper in _MODS:
            mods |= _MODS[upper]
        elif len(part) == 1 and part.isalnum():
            key = ord(part.upper())
        elif upper.startswith("F") and upper[1:].isdigit() and 1 <= int(upper[1:]) <= 24:
            key = 0x70 + int(upper[1:]) - 1
        else:
            raise ValueError(f"unknown hotkey part: {part!r}")
    if key is None:
        raise ValueError(f"hotkey has no key: {text!r}")
    return mods, key


class HotkeyManager:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._running = False
        self._ready = threading.Event()
        self._bindings: dict[str, tuple[int, int]] = {}
        self._callback: Callable[[str], None] | None = None

    def register(self, mapping: dict[str, int | str], callback: Callable[[str], None]) -> None:
        self._bindings = {name: parse_binding(spec) for name, spec in mapping.items()}
        self._callback = callback
        self._running = True
        self._ready.clear()
        self._thread = threading.Thread(target=self._loop, name="hotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(1.0)

    def _loop(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        registered: dict[int, str] = {}
        for index, (name, (mods, vk)) in enumerate(self._bindings.items(), start=1):
            if user32.RegisterHotKey(None, index, mods | MOD_NOREPEAT, vk):
                registered[index] = name
            else:
                log.warning("hotkey %r could not be registered (already in use?)", name)
        self._ready.set()

        message = wintypes.MSG()
        while self._running and user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            if message.message == WM_HOTKEY:
                name = registered.get(int(message.wParam))
                if name and self._callback is not None:
                    self._callback(name)
        for index in registered:
            user32.UnregisterHotKey(None, index)

    def unregister_all(self) -> None:
        self._running = False
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(1.0)
            self._thread = None
