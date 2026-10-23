"""Application entry point: wires capture, OCR, translation, overlay and hotkeys."""

from __future__ import annotations

import argparse
import logging
import queue
import sys
from dataclasses import asdict
from pathlib import Path

import requests
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from .capture import CaptureManager, DxgiBackend
from .config import Config, load_config, save_config
from .exclusion import AffinityGate
from .geometry import Rect
from .hotkeys import HotkeyManager
from .logging_setup import setup_logging
from .ocr import RapidOcrEngine
from .online import DeepLEngine, GeminiEngine, GoogleEngine, MyMemoryEngine, OnlineRouter
from .overlay import OverlayWindow
from .pipeline import Pipeline
from .regions import RegionSet, from_physical
from .track import TemporalTracker
from .translate import Envit5Engine, NllbEngine, TranslationCache, TranslationService
from .ui.picker import pick_regions
from .ui.settings import SettingsDialog

log = logging.getLogger(__name__)

DEFAULT_KEYS = {
    "toggle": "Ctrl+Alt+Z",
    "region": "Ctrl+Alt+R",
    "mode": "Ctrl+Alt+M",
    "hud": "Ctrl+Alt+H",
    "settings": "Ctrl+Alt+S",
    "quit": "Ctrl+Alt+Q",
}
LOCAL_REPOS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "envit5": "VietAI/envit5-translation",
}
MODES = ["single", "multi", "monitor"]


def build_local(name: str):
    from huggingface_hub import snapshot_download

    engine = NllbEngine if name == "nllb" else Envit5Engine
    local = engine(Path(snapshot_download(LOCAL_REPOS[name])))
    local.load()
    return local


def build_engine(cfg: Config):
    if cfg.engine in LOCAL_REPOS:
        return build_local(cfg.engine)
    if not cfg.online_consent:
        log.warning("online engine selected without consent; falling back to nllb")
        return build_local("nllb")

    session = requests.Session()
    if cfg.engine == "google":
        primary = GoogleEngine(session)
    elif cfg.engine == "mymemory":
        primary = MyMemoryEngine(session)
    elif cfg.engine == "deepl":
        primary = DeepLEngine(cfg.keys.get("deepl", ""), session)
    else:
        primary = GeminiEngine(cfg.keys.get("gemini", ""), session)
    fallback = build_local("nllb") if cfg.fallback_to_local else None
    return OnlineRouter(primary, fallback)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="start, run 4 s, quit")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    app = QApplication(sys.argv)

    screen_q = app.primaryScreen().geometry()
    screen = Rect(screen_q.x(), screen_q.y(), screen_q.width(), screen_q.height())

    events: queue.Queue = queue.Queue()
    overlay = OverlayWindow(screen, cfg, events)
    service = TranslationService(build_engine(cfg), TranslationCache())
    capture = CaptureManager(DxgiBackend())
    pipeline = Pipeline(
        capture,
        RapidOcrEngine(),
        TemporalTracker(cfg.stability_profile),
        service,
        events.put,
        gate=AffinityGate(),
        fps=cfg.capture_fps,
    )
    rects = RegionSet.from_config(cfg, screen)
    pipeline.set_regions(rects)
    overlay.set_regions(rects)
    pipeline.start()

    actions: queue.Queue = queue.Queue()
    hotkeys = HotkeyManager()

    def handle(action: str) -> None:
        if action == "quit":
            shutdown()
            app.quit()
        elif action == "toggle":
            pipeline.set_enabled(not pipeline.enabled)
            if not pipeline.enabled:
                overlay.clear()
        elif action == "region":
            overlay.hide()
            picked = pick_regions(screen)
            overlay.show()
            if picked:
                pipeline.set_regions(picked)
                overlay.set_regions(picked)
                cfg.regions = [asdict(from_physical(rect, screen)) for rect in picked]
                save_config(cfg)
        elif action == "mode":
            cfg.region_mode = MODES[(MODES.index(cfg.region_mode) + 1) % len(MODES)]
            save_config(cfg)
            new_rects = RegionSet.from_config(cfg, screen)
            pipeline.set_regions(new_rects)
            overlay.set_regions(new_rects)
        elif action == "settings":
            dialog = SettingsDialog(cfg, apply_settings)
            dialog.exec()
        elif action == "hud":
            pass  # HUD is task 14

    def apply_settings(new_cfg: Config) -> None:
        pipeline.fps = new_cfg.capture_fps
        pipeline.set_engine(build_engine(new_cfg))
        new_rects = RegionSet.from_config(new_cfg, screen)
        pipeline.set_regions(new_rects)
        overlay.set_regions(new_rects)

    def shutdown() -> None:
        hotkeys.unregister_all()
        pipeline.stop()
        overlay.close()
        capture.close()

    try:
        hotkeys.register(cfg.keys or DEFAULT_KEYS, actions.put)
    except Exception:  # noqa: BLE001 - hotkeys are optional
        log.exception("hotkey registration failed")

    def drain() -> None:
        while True:
            try:
                action = actions.get_nowait()
            except queue.Empty:
                return
            handle(action)

    timer = QTimer()
    timer.timeout.connect(drain)
    timer.start(50)
    app.aboutToQuit.connect(shutdown)

    if args.smoke:
        def finish() -> None:
            print("stats:", pipeline.stats())
            shutdown()
            app.quit()

        QTimer.singleShot(4000, finish)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
