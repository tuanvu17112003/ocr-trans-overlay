"""Application entry point: control window, capture, OCR, translation, overlay."""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import time
from dataclasses import asdict

import requests
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

from .blocks import BlockMemory
from .capture import CaptureManager, DxgiBackend, WgcBackend
from .config import Config, load_config, save_config
from .exclusion import AffinityGate
from .geometry import Rect
from .hotkeys import HotkeyManager
from .hud import HudWindow
from .logging_setup import setup_logging
from .ocr import RapidOcrEngine
from .online import DeepLEngine, GeminiEngine, GoogleEngine, MyMemoryEngine, OnlineRouter
from .overlay import OverlayWindow
from .pipeline import Pipeline
from .regions import (
    NormalizedRegion,
    RegionSet,
    add_region,
    remove_region,
    replace_region,
    to_physical,
)
from .scan import ScanBudget
from .translate import Envit5Engine, NllbEngine, TranslationCache, TranslationService
from .ui.control import ControlWindow
from .ui.outline import RegionOutline
from .ui.picker import adjust_rect, pick_regions
from .ui.theme import apply as apply_theme

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
MODES = ["single", "multi"]


def build_local(name: str):
    from huggingface_hub import snapshot_download

    engine = NllbEngine if name == "nllb" else Envit5Engine
    local = engine(snapshot_download(LOCAL_REPOS[name]))
    local.load()
    return local


def build_engine(cfg: Config):
    if cfg.engine in LOCAL_REPOS:
        return build_local(cfg.engine)
    if cfg.engine == "argos":
        from .argos import ArgosEngine

        engine = ArgosEngine()
        engine.load()
        return engine
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
    parser.add_argument("--smoke", action="store_true", help="start, run 8 s, quit")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    app = QApplication(sys.argv)
    apply_theme(app)

    screen_q = app.primaryScreen().geometry()
    screen = Rect(screen_q.x(), screen_q.y(), screen_q.width(), screen_q.height())

    events: queue.Queue = queue.Queue()
    overlay = OverlayWindow(screen, cfg, events)
    overlay.hide()
    outline = RegionOutline(screen)

    if os.environ.get("OCRTRANS_DXGI"):
        capture = CaptureManager(DxgiBackend(), WgcBackend(monitor_index=1))
    else:
        # WGC first: an active Desktop Duplication session keeps the output out
        # of the video's independent-flip path and measurably drops frames
        # (5-8 per 15 s at 1440p60 in Edge; WGC dropped 0 in the same runs).
        capture = CaptureManager(WgcBackend(monitor_index=1), DxgiBackend())
    log.info(
        "capture %s (backup %s); overlay %s; debug %s; region %s",
        type(capture.main).__name__,
        type(capture.fallback).__name__,
        "off" if os.environ.get("OCRTRANS_NO_OVERLAY") else "on",
        "on" if os.environ.get("OCRTRANS_DEBUG") else "off",
        RegionSet.from_config(cfg, screen),
    )
    state: dict = {
        "pipeline": None,
        "engine_name": None,
        "running": False,
        "starting": False,
        "picking": False,
        "error": "",
    }
    actions: queue.Queue = queue.Queue()
    hotkeys = HotkeyManager()
    hud_state = {"visible": False}

    _logged_at = {"at": 0.0}

    def status() -> dict:
        pipeline = state["pipeline"]
        if pipeline is None:
            return {
                "engine": cfg.engine,
                "health": "-",
                "p50": "-",
                "p95": "-",
                "ocr_passes": 0,
                "ocr_errors": 0,
                "mt_errors": 0,
                "blocks": 0,
                "error": state["error"],
            }
        data = {
            **pipeline.stats(),
            "engine": state["engine_name"] or cfg.engine,
            "health": capture.health.name,
            "error": state["error"],
            "rendered": overlay.shown,
            "hidden_stale": overlay.state.hidden_stale,
        }
        now = time.monotonic()
        if now - _logged_at["at"] >= 5.0:
            _logged_at["at"] = now
            log.info("stats %s", data)
        return data

    hud = HudWindow(status)

    def load_engine(config: Config):
        """Build and load a model, showing progress without re-entering start()."""
        window.set_status_text("loading model...")
        window.repaint()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            engine = build_engine(config)
        except Exception as exc:  # noqa: BLE001 - a failed load must not kill the app
            log.exception("model load failed")
            state["error"] = f"load failed: {exc}"
            return None
        finally:
            QApplication.restoreOverrideCursor()
        state["error"] = ""
        return engine

    def start() -> None:
        if state["starting"]:
            return
        state["starting"] = True
        try:
            _start()
        finally:
            state["starting"] = False

    def _start() -> None:
        window.read_config()
        save_config(cfg)
        if not cfg.regions:
            state["error"] = "add a region first"
            window.set_running(False)
            window.set_status_text(state["error"])
            return
        pipeline = state["pipeline"]
        if pipeline is None:
            engine = load_engine(cfg)
            if engine is None:
                window.set_running(False)
                return
            pipeline = Pipeline(
                capture,
                RapidOcrEngine(),
                TranslationService(engine, TranslationCache()),
                events.put,
                memory=BlockMemory(cfg.stability_profile),
                budget=ScanBudget(cfg.capture_fps),
                gate=AffinityGate(),
                fps=cfg.capture_fps,
            )
            rects = RegionSet.from_config(cfg, screen)
            pipeline.set_regions(rects)
            overlay.set_regions(rects)
            pipeline.start()
            state["pipeline"] = pipeline
            state["engine_name"] = engine.name
        else:
            if state["engine_name"] != cfg.engine:
                # free the old model before the new one allocates VRAM
                pipeline.set_enabled(False)
                if not pipeline.unload_engine():
                    state["error"] = "translation still finishing; press Start again"
                    window.set_running(False)
                    return
                state["engine_name"] = None
                engine = load_engine(cfg)
                if engine is None:
                    window.set_running(False)
                    return
                pipeline.adopt_engine(engine)
                state["engine_name"] = engine.name
            pipeline.set_fps(cfg.capture_fps)
            pipeline.set_profile(cfg.stability_profile)
            rects = RegionSet.from_config(cfg, screen)
            pipeline.set_regions(rects)
            overlay.set_regions(rects)
            pipeline.set_enabled(True)
        if not os.environ.get("OCRTRANS_NO_OVERLAY"):
            overlay.show()
        state["running"] = True
        window.set_running(True)
        outline_visibility()

    def stop() -> None:
        pipeline = state["pipeline"]
        if pipeline is not None:
            pipeline.set_enabled(False)
        overlay.clear()
        overlay.hide()
        state["running"] = False
        window.set_running(False)
        refresh_outline()
        outline.set_visible_for(stopped=True, force=window.show_outlines.isChecked())

    def refresh_outline() -> None:
        if not cfg.regions:
            outline.set_regions([], None, set())
            return
        rects = [to_physical(NormalizedRegion(**raw), screen) for raw in cfg.regions]
        applied = 1 if window.mode.currentText() == "single" else len(rects)
        selected = window.selected_region()
        active = selected if selected is not None and selected < len(rects) else None
        outline.set_regions(rects, active, set(range(applied, len(rects))))

    def outline_visibility() -> None:
        outline.set_visible_for(
            stopped=not state["running"], force=window.show_outlines.isChecked()
        )

    def apply_regions() -> None:
        save_config(cfg)
        rects = RegionSet.from_config(cfg, screen)
        pipeline = state["pipeline"]
        if pipeline is not None:
            pipeline.set_regions(rects)
        overlay.set_regions(rects)
        window.refresh_regions()
        refresh_outline()

    def saved_regions() -> list[NormalizedRegion]:
        return [NormalizedRegion(**raw) for raw in cfg.regions]

    def pick() -> None:
        if state["picking"]:
            return
        state["picking"] = True
        try:
            overlay.hide()
            outline.hide()
            picked = pick_regions(screen)
        finally:
            state["picking"] = False
        if state["running"]:
            if not os.environ.get("OCRTRANS_NO_OVERLAY"):
                overlay.show()
        if picked:
            cfg.regions = [
                asdict(region) for region in add_region(saved_regions(), picked[0], screen)
            ]
            apply_regions()
        outline_visibility()

    def adjust_region(index: int) -> None:
        if not 0 <= index < len(cfg.regions) or state["picking"]:
            return
        state["picking"] = True
        try:
            overlay.hide()
            outline.hide()
            current = to_physical(NormalizedRegion(**cfg.regions[index]), screen)
            edited = adjust_rect(screen, current)
        finally:
            state["picking"] = False
        if state["running"]:
            if not os.environ.get("OCRTRANS_NO_OVERLAY"):
                overlay.show()
        if edited is not None:
            cfg.regions = [
                asdict(region)
                for region in replace_region(saved_regions(), index, edited, screen)
            ]
            apply_regions()
        outline_visibility()

    def delete_region(index: int) -> None:
        cfg.regions = [asdict(region) for region in remove_region(saved_regions(), index)]
        apply_regions()

    def apply_settings(new_cfg: Config) -> None:
        save_config(new_cfg)
        overlay.update()
        pipeline = state["pipeline"]
        if pipeline is not None:
            pipeline.set_fps(new_cfg.capture_fps)
            pipeline.set_profile(new_cfg.stability_profile)

    def cycle_mode() -> None:
        cfg.region_mode = MODES[(MODES.index(cfg.region_mode) + 1) % len(MODES)]
        save_config(cfg)
        window.mode.setCurrentText(cfg.region_mode)
        rects = RegionSet.from_config(cfg, screen)
        pipeline = state["pipeline"]
        if pipeline is not None:
            pipeline.set_regions(rects)
        overlay.set_regions(rects)
        window.refresh_regions()
        refresh_outline()

    def shutdown() -> None:
        hotkeys.unregister_all()
        pipeline = state["pipeline"]
        if pipeline is not None:
            pipeline.stop()
        overlay.close()
        outline.close()
        hud.close()
        capture.close()

    def handle(action: str) -> None:
        if state["picking"] and action in {"region", "toggle", "mode", "settings"}:
            return
        if action == "quit":
            app.quit()
        elif action == "toggle":
            stop() if state["running"] else start()
        elif action == "region":
            pick()
        elif action == "mode":
            cycle_mode()
        elif action == "hud":
            hud_state["visible"] = not hud_state["visible"]
            hud.set_visible(hud_state["visible"])
        elif action == "settings":
            window.show()
            window.raise_()

    window = ControlWindow(
        cfg,
        start,
        stop,
        pick,
        adjust_region,
        delete_region,
        apply_settings,
        status,
        screen,
        on_close=app.quit,
    )
    window.show()
    window.region_list.currentRowChanged.connect(lambda _row: refresh_outline())
    window.mode.currentTextChanged.connect(lambda _mode: refresh_outline())
    window.show_outlines.toggled.connect(lambda _checked: outline_visibility())
    refresh_outline()
    outline.set_visible_for(stopped=True, force=window.show_outlines.isChecked())

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
            pipeline = state["pipeline"]
            print("stats:", None if pipeline is None else pipeline.stats())
            app.quit()

        def smoke_start() -> None:
            start()
            QTimer.singleShot(8000, finish)

        QTimer.singleShot(0, smoke_start)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

