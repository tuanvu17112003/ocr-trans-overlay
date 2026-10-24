"""End-to-end: an empty screen must not spin OCR, and text must be picked up again."""

import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocrtrans.blocks import BlockMemory
from ocrtrans.capture import Health
from ocrtrans.geometry import Rect
from ocrtrans.ocr import RapidOcrEngine
from ocrtrans.pipeline import EventKind, Pipeline
from ocrtrans.scan import ScanBudget
from ocrtrans.translate import TranslationCache, TranslationService


class SwitchCapture:
    def __init__(self):
        self.frame = np.zeros((200, 800, 3), np.uint8)

    def grab(self, rect):
        return self.frame.copy(), Health.OK


class FakeEngine:
    name, kind = "fake", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return [f"VI {text}" for text in texts]


def text_frame() -> np.ndarray:
    image = Image.new("RGB", (800, 200), (10, 10, 10))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 34)
    draw.text((30, 70), "Open the gate, traveler.", font=font, fill=(240, 240, 240))
    return np.array(image)[:, :, ::-1].copy()


def test_empty_screen_does_not_spin_ocr_and_recovers_when_text_appears():
    cap = SwitchCapture()
    events = []
    ocr = RapidOcrEngine()
    ocr.read(text_frame())  # model load is a one-time startup cost, not the scan budget
    p = Pipeline(
        cap,
        ocr,
        TranslationService(FakeEngine(), TranslationCache()),
        events.append,
        memory=BlockMemory("safe"),
        budget=ScanBudget(15),
        fps=15,
    )
    p.set_regions([Rect(0, 0, 800, 200)])
    p.start()
    time.sleep(1.5)
    empty_passes = p.stats()["ocr_passes"]
    assert empty_passes <= 4, f"OCR spun {empty_passes} times on a static empty screen"
    assert p.stats()["blocks"] == 0

    cap.frame = text_frame()
    deadline = time.monotonic() + 10.0
    translated = False
    while time.monotonic() < deadline and not translated:
        time.sleep(0.1)
        translated = any(event.kind is EventKind.TRANSLATION_RESULT for event in events)
    p.stop()

    assert translated, "text appearing on screen was never translated"
    assert p.stats()["ocr_passes"] > empty_passes
