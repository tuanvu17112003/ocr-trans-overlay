"""Opt-in tests against the running Gunfire Reborn window (WGC + real OCR + NLLB)."""

from __future__ import annotations

import ctypes
import queue
import re
import time
from ctypes import wintypes
from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.real

TITLE_PART = "gunfire"


@dataclass
class GameWindow:
    hwnd: int
    title: str


def find_game_window() -> GameWindow | None:
    user32 = ctypes.windll.user32
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if TITLE_PART in title.lower() and "launcher" not in title.lower():
                found.append((hwnd, title))
        return True

    user32.EnumWindows(callback, 0)
    return GameWindow(*found[0]) if found else None


@pytest.fixture(scope="session")
def game_window() -> GameWindow:
    window = find_game_window()
    if window is None:
        pytest.skip("Gunfire Reborn window not found; start it in borderless mode")
    if ctypes.windll.user32.IsIconic(window.hwnd):
        pytest.skip("game window is minimized")
    return window


@pytest.fixture(scope="session")
def game_frame(game_window: GameWindow):
    from windows_capture import WindowsCapture

    frames = []
    try:
        capture = WindowsCapture(
            window_hwnd=game_window.hwnd, cursor_capture=False, draw_border=False
        )
    except Exception as exc:  # noqa: BLE001 - exclusive fullscreen must skip, not fail
        pytest.skip(f"WGC window capture failed: {exc}")

    @capture.event
    def on_frame_arrived(frame, control):
        frames.append(frame.frame_buffer.copy())
        control.stop()

    @capture.event
    def on_closed():
        pass

    try:
        control = capture.start_free_threaded()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"WGC window capture failed: {exc}")

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not frames:
        time.sleep(0.05)
    try:
        control.stop()
    except Exception:  # noqa: BLE001 - already stopped by the callback
        pass
    if not frames:
        pytest.skip("WGC window capture produced no frame (exclusive fullscreen?)")
    return frames[-1]


def test_game_window_is_found(game_window: GameWindow):
    assert game_window.hwnd > 0
    assert TITLE_PART in game_window.title.lower()


def test_capture_game_window(game_frame):
    assert game_frame is not None
    assert game_frame.shape[0] > 100 and game_frame.shape[1] > 100
    assert game_frame.shape[2] in (3, 4)
    assert float(game_frame.mean()) > 2.0


def test_ocr_game_frame(game_frame):
    from ocrtrans.ocr import RapidOcrEngine

    started = time.perf_counter()
    boxes = RapidOcrEngine().read(game_frame[:, :, :3])
    elapsed = (time.perf_counter() - started) * 1000
    if not boxes:
        pytest.skip("no text on the current game screen")
    print(f"ocr: {len(boxes)} boxes in {elapsed:.0f} ms")
    assert len(boxes) > 0


def test_translate_game_text(game_frame):
    from ocrtrans.app import build_local
    from ocrtrans.ocr import RapidOcrEngine, group_blocks

    boxes = RapidOcrEngine().read(game_frame[:, :, :3])
    blocks = group_blocks(boxes)
    if not blocks:
        pytest.skip("no text on the current game screen")
    texts = [block.text for block in blocks[:5]]
    engine = build_local("nllb")
    started = time.perf_counter()
    out = engine.translate(texts)
    elapsed = time.perf_counter() - started
    assert len(out) == len(texts)
    assert all(text.strip() for text in out)
    for source, translated in zip(texts, out):
        assert set(re.findall(r"\d+", source)) <= set(re.findall(r"\d+", translated))
    print(f"translate: {len(texts)} blocks in {elapsed * 1000:.0f} ms -> {out[:2]}")


def test_pipeline_against_game(game_window: GameWindow, game_frame):
    from ocrtrans.app import build_local
    from ocrtrans.blocks import BlockMemory
    from ocrtrans.capture import CaptureManager, DxgiBackend
    from ocrtrans.geometry import Rect
    from ocrtrans.ocr import RapidOcrEngine
    from ocrtrans.pipeline import Pipeline
    from ocrtrans.scan import ScanBudget
    from ocrtrans.translate import TranslationCache, TranslationService

    height, width = game_frame.shape[:2]
    events: queue.Queue = queue.Queue()
    backend = DxgiBackend()
    pipeline = Pipeline(
        CaptureManager(backend),
        RapidOcrEngine(),
        TranslationService(build_local("nllb"), TranslationCache()),
        events.put,
        memory=BlockMemory("safe"),
        budget=ScanBudget(10),
        fps=10,
    )
    pipeline.set_regions([Rect(0, 0, width, height)])
    pipeline.start()
    try:
        time.sleep(15)
        stats = pipeline.stats()
        alive = all(thread.is_alive() for thread in pipeline._threads)
    finally:
        pipeline.stop()
        backend.close()
    assert alive, "a pipeline thread died"
    if stats["ocr_passes"] == 0:
        pytest.skip("no OCR pass reached the screen (exclusive fullscreen or black frames)")
    print("pipeline:", stats)
