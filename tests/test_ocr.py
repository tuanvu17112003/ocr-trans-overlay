import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from ocrtrans.geometry import Box, Rect
from ocrtrans.ocr import RapidOcrEngine, group_blocks


def test_lines_of_one_paragraph_group_in_reading_order():
    boxes = [
        Box(Rect(10, 34, 200, 20), "second line", 0.9),
        Box(Rect(10, 10, 200, 20), "first line", 0.9),
        Box(Rect(10, 200, 200, 20), "menu", 0.9),
    ]
    blocks = group_blocks(boxes)
    assert [b.text for b in blocks] == ["first line second line", "menu"]


def test_confidence_filter_drops_noise():
    boxes = [Box(Rect(0, 0, 50, 12), "x", 0.2)]
    assert group_blocks(boxes, min_confidence=0.6) == []


@pytest.mark.gpu
def test_gpu_read_recovers_rendered_sentence():
    img = Image.new("RGB", (800, 200), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 44)
    draw.text((30, 70), "Open the gate, traveler.", font=font, fill=(240, 240, 240))
    frame = np.array(img)[:, :, ::-1].copy()

    engine = RapidOcrEngine(use_cuda=True)
    boxes = engine.read(frame)
    text = " ".join(b.text for b in boxes).lower()
    assert "gate" in text and "traveler" in text
