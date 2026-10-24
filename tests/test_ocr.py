import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from ocrtrans.geometry import Box, Rect
from ocrtrans.ocr import RapidOcrEngine, group_blocks, merge_boxes, plan_tiles


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


def test_a_short_string_in_a_huge_box_is_dropped():
    # Real log: a moving video made the detector report one 1059x317 box.
    boxes = [Box(Rect(0, 0, 1059, 317), "TANOV", 0.95)]
    assert group_blocks(boxes) == []


def test_a_real_subtitle_line_in_the_same_region_is_kept():
    boxes = [Box(Rect(0, 0, 757, 79), "So this is a very pretty phone.", 0.95)]
    blocks = group_blocks(boxes)
    assert [b.text for b in blocks] == ["So this is a very pretty phone."]


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


def test_centred_wrapped_caption_is_one_block():
    # Geometry from a real game subtitle: two centred lines of one sentence,
    # the second indented by more than two line heights.
    boxes = [
        Box(
            Rect(0, 0, 1240, 46),
            "Not what you would expect from a Sentinel, is it? But that's exactly",
            0.95,
        ),
        Box(
            Rect(160, 58, 920, 46),
            "why, through all this, we haven't lost ourselves.",
            0.95,
        ),
    ]
    blocks = group_blocks(boxes)
    assert len(blocks) == 1
    assert "lost ourselves" in blocks[0].text


def test_a_speaker_name_is_not_merged_into_the_dialogue():
    # Measured boxes (scripts/ocr_layout_samples.py, case A): name 50 px,
    # dialogue 40 px, 26 px apart.
    boxes = [
        Box(Rect(261, 26, 455, 50), "Beleaguered Celestial", 1.0),
        Box(Rect(228, 102, 522, 40), "Oh, what am I supposed to do now...", 0.986),
    ]
    blocks = group_blocks(boxes)
    assert [b.text for b in blocks] == [
        "Beleaguered Celestial",
        "Oh, what am I supposed to do now...",
    ]


def test_a_same_size_speaker_name_is_still_a_label():
    # Measured case D: name and dialogue are both 40 px tall (3 px gap).
    boxes = [
        Box(Rect(295, 27, 384, 40), "Beleaguered Celestial", 1.0),
        Box(Rect(228, 75, 522, 40), "Oh, what am I supposed to do now..", 0.989),
    ]
    blocks = group_blocks(boxes)
    assert [b.text for b in blocks] == [
        "Beleaguered Celestial",
        "Oh, what am I supposed to do now..",
    ]


def test_a_speaker_name_over_a_two_line_dialogue_splits_off():
    # Measured case G: 49 px name, two dialogue lines below.
    boxes = [
        Box(Rect(260, 8, 455, 49), "Beleaguered Celestial", 1.0),
        Box(Rect(228, 69, 523, 41), "Oh, what am I supposed to do now..", 0.976),
        Box(Rect(346, 106, 284, 45), "Nothing left to say.", 0.998),
    ]
    blocks = group_blocks(boxes)
    assert [b.text for b in blocks] == [
        "Beleaguered Celestial",
        "Oh, what am I supposed to do now.. Nothing left to say.",
    ]


def test_a_wrapped_two_line_caption_stays_one_block():
    # Measured case E: centred lines, no label shape on the first line.
    boxes = [
        Box(Rect(139, 35, 700, 36), "Not what you would expect from a Sentinel, is it?", 0.988),
        Box(
            Rect(210, 78, 555, 37),
            "But that's exactly why, through all this,",
            1.0,
        ),
    ]
    blocks = group_blocks(boxes)
    assert len(blocks) == 1
    assert "through all this" in blocks[0].text


def test_a_short_first_line_still_joins_its_continuation():
    boxes = [
        Box(Rect(0, 0, 300, 40), "I can't", 0.95),
        Box(Rect(0, 52, 300, 40), "believe it.", 0.95),
    ]
    blocks = group_blocks(boxes)
    assert len(blocks) == 1
    assert blocks[0].text == "I can't believe it."


def test_a_terminal_punctuation_short_line_is_not_a_label():
    boxes = [
        Box(Rect(0, 0, 200, 40), "The End.", 0.95),
        Box(Rect(0, 48, 300, 40), "We move on.", 0.95),
    ]
    assert len(group_blocks(boxes)) == 1


def test_plan_tiles_covers_a_2560x1440_region_in_at_most_four_tiles():
    tiles = plan_tiles(2560, 1440)
    assert 1 < len(tiles) <= 4
    assert all(t.w <= 1344 and t.h <= 1344 for t in tiles)
    assert tiles[0].x == 0 and tiles[0].y == 0
    assert max(t.right for t in tiles) == 2560
    assert max(t.bottom for t in tiles) == 1440


def test_merge_boxes_keeps_the_higher_confidence_duplicate():
    a = Box(Rect(0, 0, 100, 20), "HELLO", 0.7)
    b = Box(Rect(2, 1, 100, 20), "HELLO", 0.9)
    merged = merge_boxes([a, b])
    assert len(merged) == 1 and merged[0].confidence == 0.9


def test_cpu_fallback_caps_the_onnx_thread_count():
    params = RapidOcrEngine._params(False)
    assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == 2
    assert params["EngineConfig.onnxruntime.inter_op_num_threads"] == 1


def test_gpu_params_keep_the_same_thread_cap():
    params = RapidOcrEngine._params(True)
    assert params["EngineConfig.onnxruntime.use_cuda"] is True
    assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == 2


def test_a_music_note_read_as_a_letter_is_dropped():
    from ocrtrans.ocr import _trim_edge_noise

    assert _trim_edge_noise("S HEART MADE OF GLASS S") == "HEART MADE OF GLASS"
    assert _trim_edge_noise("J SKIN TO BONE") == "SKIN TO BONE"
    assert _trim_edge_noise("A tale of two") == "A tale of two"
    assert _trim_edge_noise("I am here") == "I am here"
    assert _trim_edge_noise("S") == "S"


def test_a_quoted_dialogue_does_not_hide_the_name():
    # Measured case M: the recogniser reads the opening quote as a broken glyph,
    # so the uppercase test must look at the first letter, not the first char.
    boxes = [
        Box(Rect(261, 23, 455, 49), "Beleaguered Celestial", 1.0),
        Box(Rect(214, 83, 540, 43), "\ufffdOh, what am I supposed to do now...'", 0.97),
    ]
    blocks = group_blocks(boxes)
    assert [b.text for b in blocks] == [
        "Beleaguered Celestial",
        "\ufffdOh, what am I supposed to do now...'",
    ]


def test_a_five_word_name_is_still_a_label():
    # Measured case N: capitalisation ratio, not every word.
    boxes = [
        Box(Rect(267, 30, 444, 39), "High Priestess of the Sun", 0.989),
        Box(Rect(314, 88, 350, 38), "The gate opens at dawn.", 1.0),
    ]
    assert len(group_blocks(boxes)) == 2


def test_two_columns_merge_within_their_own_column():
    # Measured case I: a quest tracker on the left and a resource panel on the
    # right; each column is one paragraph, and they never join.
    boxes = [
        Box(Rect(34, 39, 223, 37), "Quest: find the key", 0.976),
        Box(Rect(694, 41, 93, 32), "HP 100", 1.0),
        Box(Rect(35, 79, 211, 34), "Reward: 300 gold", 1.0),
        Box(Rect(695, 80, 120, 31), "Ammo 30", 1.0),
    ]
    blocks = group_blocks(boxes)
    assert [b.text for b in blocks] == [
        "Quest: find the key Reward: 300 gold",
        "HP 100 Ammo 30",
    ]
