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
def test_downscale_reads_full_width_and_maps_boxes():
    img = Image.new("RGB", (2000, 200), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 44)
    draw.text((1000, 70), "Open the gate", font=font, fill=(240, 240, 240))
    frame = np.array(img)[:, :, ::-1].copy()

    engine = RapidOcrEngine(use_cuda=True, downscale_width=1000)
    boxes = engine.read(frame)
    text = " ".join(b.text for b in boxes).lower()
    assert "gate" in text
    assert max(b.rect.x for b in boxes) > 900


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


def test_grouping_stops_at_six_lines():
    boxes = [Box(Rect(10, 10 + i * 18, 200, 16), f"line {i}", 0.9) for i in range(8)]
    blocks = group_blocks(boxes)
    assert len(blocks) == 2
    assert len(blocks[0].lines) == 6


def test_same_row_labels_are_not_merged():
    boxes = [
        Box(Rect(10, 100, 30, 22), "Yes", 0.9),
        Box(Rect(50, 100, 30, 22), "No", 0.9),
    ]
    assert len(group_blocks(boxes)) == 2


def test_five_line_paragraph_stays_one_block():
    boxes = [Box(Rect(10, 10 + index * 18, 300, 16), f"line {index}", 0.9) for index in range(5)]
    blocks = group_blocks(boxes)
    assert len(blocks) == 1
    assert len(blocks[0].lines) == 5



def test_tight_lines_do_not_interleave_groups():
    boxes = [Box(Rect(10, 10 + index * 15, 200, 20), f"line {index}", 0.9) for index in range(4)]
    blocks = group_blocks(boxes)
    assert len(blocks) == 4
    assert all(len(block.lines) == 1 for block in blocks)


def test_split_line_is_one_block():
    boxes = [Box(Rect(10, 100, 50, 22), "Open", 0.9), Box(Rect(64, 100, 80, 22), "the gate", 0.9)]
    blocks = group_blocks(boxes)
    assert len(blocks) == 1
    assert blocks[0].text == "Open the gate"



