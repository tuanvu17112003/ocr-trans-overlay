"""End-to-end scenario on this machine.

Paints a caption where the app is configured to look, runs the real app with
--smoke and debug logging, then asserts the grouping and the translation from
the app's own log. Nothing is mocked: WGC capture, RapidOCR, the configured
translation engine and the overlay all run.

Run: .venv/Scripts/python scripts/scenario_check.py
     .venv/Scripts/python scripts/scenario_check.py --keep-open

Each scene takes ~12 s (model load + 8 s of scanning).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

NAME_TEXT = "Beleaguered Celestial"
DIALOGUE_START = "Oh, what am I supposed to do now"
WRAP_1 = "Not what you would expect from a Sentinel, is it?"
WRAP_2 = "But that's exactly why, through all this,"
LONG_TEXT = "Through all this we have not lost ourselves, and we never will."
OCR_PREFIX = "ocr text: "


# ---------------------------------------------------------------- scene side
def scene_image(name: str, size: tuple[int, int]):
    from PIL import Image, ImageFont

    from ocr_layout_samples import case_title_dialogue, case_wrapped, canvas, centered

    if name == "name":
        base = case_title_dialogue(30)
    elif name == "wrapped":
        base = case_wrapped(2)
    else:
        image, draw = canvas()
        centered(draw, LONG_TEXT, ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 28), 80)
        base = image
    width, height = size
    canvas_image = Image.new("RGB", (width, height), (44, 35, 30))
    canvas_image.paste(base, (max(0, (width - base.width) // 2), max(0, (height - base.height) // 2)))
    return canvas_image


def run_scene(name: str, region: tuple[int, int, int, int]) -> None:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import QApplication, QLabel

    app = QApplication([])
    image = scene_image(name, (region[2], region[3])).convert("RGB")
    data = image.tobytes("raw", "RGB")
    qimage = QImage(data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888)
    label = QLabel()
    label.setPixmap(QPixmap.fromImage(qimage))
    label.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
    label.setGeometry(*region)
    label.show()
    # Any other window that pops up must not cover the scene mid-run.
    keep_on_top = QTimer()
    keep_on_top.timeout.connect(label.raise_)
    keep_on_top.start(300)
    app.exec()


# ------------------------------------------------------------- checker side
def config_region() -> tuple[int, int, int, int]:
    from PyQt6.QtWidgets import QApplication

    from ocrtrans.config import load_config
    from ocrtrans.geometry import Rect
    from ocrtrans.regions import RegionSet

    app = QApplication.instance() or QApplication([])
    screen_q = app.primaryScreen().geometry()
    screen = Rect(screen_q.x(), screen_q.y(), screen_q.width(), screen_q.height())
    rect = RegionSet.from_config(load_config(), screen)[0]
    return rect.x, rect.y, rect.w, rect.h


def log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "OCRTrans"
    return base / "app.log"


def run_scene_process(name: str, region: tuple[int, int, int, int]) -> subprocess.Popen:
    env = {**os.environ, "SCENE_REGION": json.dumps(list(region))}
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--scene", name],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_app() -> subprocess.Popen:
    env = {**os.environ, "OCRTRANS_DEBUG": "1"}
    return subprocess.Popen(
        [sys.executable, "-m", "ocrtrans.app", "--smoke"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def read_new_log(offset: int) -> list[str]:
    path = log_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        return handle.read().splitlines()


def parsed_scans(lines: list[str]) -> list[list[tuple[int, int, str]]]:
    """Rebuild the (w, h, text) blocks of every scan from the debug lines."""
    scans: list[list[tuple[int, int, str]]] = []
    for line in lines:
        index = line.find(OCR_PREFIX)
        if index < 0:
            continue
        try:
            entries = ast.literal_eval(line[index + len(OCR_PREFIX):])
        except (ValueError, SyntaxError):
            continue
        scan = [
            (int(entry[0]), int(entry[1]), str(entry[2]))
            for entry in entries
            if isinstance(entry, tuple) and len(entry) == 3
        ]
        scans.append(scan)
    return scans


def check_scene(name: str, region, keep_open: bool = False) -> bool:
    path = log_path()
    offset = path.stat().st_size if path.exists() else 0
    scene = run_scene_process(name, region)
    time.sleep(2.0)  # let the scene paint before the first grab
    app = start_app()
    try:
        app.wait(timeout=45)
    finally:
        if not keep_open:
            scene.terminate()
    lines = read_new_log(offset)
    scans = parsed_scans(lines)
    print(f"\n=== scene {name} ({len(lines)} log lines, {len(scans)} scans) ===")
    return report(name, lines, scans)


def report(name: str, lines: list[str], scans: list[list[tuple[int, int, str]]]) -> bool:
    joined = "\n".join(lines)
    ok = True
    if "capture monitor" not in joined:
        print("FAIL: no 'capture monitor' line")
        ok = False
    if "truncated at the font floor" in joined:
        print("FAIL: a block was truncated")
        ok = False

    noise = sum(1 for scan in scans for _w, _h, text in scan if not _caption_text(name, text))
    if noise:
        print(f"  note: {noise} blocks from outside the caption (other windows on screen)")

    if name == "name":
        bad = [
            scan
            for scan in scans
            if any(NAME_TEXT in t and DIALOGUE_START in t for _w, _h, t in scan)
        ]
        good = [
            scan
            for scan in scans
            if any(t.strip() == NAME_TEXT for _w, _h, t in scan)
            and any(DIALOGUE_START in t for _w, _h, t in scan)
        ]
        if bad:
            print("FAIL: the speaker name is merged into the dialogue")
            print(f"  {[t for _w, _h, t in bad[0]][:4]!r}")
            ok = False
        elif good:
            print(f"PASS: {len(good)} scans saw the name and the dialogue as separate blocks")
        else:
            print("FAIL: never saw both lines in one scan")
            ok = False
    elif name == "wrapped":
        good = [
            scan
            for scan in scans
            if any(WRAP_1[:15] in t and "But that's exactly" in t for _w, _h, t in scan)
        ]
        if good:
            print(f"PASS: {len(good)} scans saw the two-line caption as one block")
        else:
            print("FAIL: the wrapped caption was split")
            for scan in scans[-1:]:
                print(f"  {[t[:50] for _w, _h, t in scan]!r}")
            ok = False
    elif name == "long":
        good = [scan for scan in scans if any(LONG_TEXT[:15] in t for _w, _h, t in scan)]
        if good:
            print(f"PASS: the long caption was read in {len(good)} scans without an ellipsis")
        else:
            print("FAIL: the long caption was not read")
            ok = False

    translations = [line.split(" mt ", 1)[1] for line in lines if " mt " in line]
    print(f"  translations: {len(translations)}")
    for translation in translations[-3:]:
        print(f"  {translation}")
    if not translations:
        print("FAIL: nothing was translated")
        ok = False

    stats = last_stats(lines)
    print(
        "  stats: health={health} grab_p50={grab_p50} ms ocr_p50={ocr_p50} ms "
        "fit_truncated={fit_truncated} rendered={rendered}".format(**stats)
        if stats
        else "  stats: none"
    )
    if stats:
        if stats.get("health") != "OK":
            print(f"FAIL: capture health is {stats.get('health')}")
            ok = False
        if stats.get("fit_truncated"):
            print("FAIL: fit_truncated is not zero")
            ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def last_stats(lines: list[str]) -> dict:
    for line in reversed(lines):
        index = line.find("stats {")
        if index < 0:
            continue
        try:
            return ast.literal_eval(line[index + len("stats "):])
        except (ValueError, SyntaxError):
            continue
    return {}


def _caption_text(name: str, text: str) -> bool:
    known = (NAME_TEXT, DIALOGUE_START, WRAP_1[:15], WRAP_2[:15], LONG_TEXT[:15])
    return any(fragment in text for fragment in known)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--scene" in sys.argv:
        region = tuple(json.loads(os.environ["SCENE_REGION"]))
        run_scene(sys.argv[sys.argv.index("--scene") + 1], region)
        return 0

    region = config_region()
    print(f"region: {region}")
    keep = "--keep-open" in sys.argv
    results = [check_scene(name, region, keep) for name in ("name", "wrapped", "long")]
    print("\nSUMMARY:", "PASS" if all(results) else "FAIL", results)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
