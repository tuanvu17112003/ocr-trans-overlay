"""Render subtitle layouts, run the real detector, print the boxes it returns.

Used to build test fixtures from measured geometry instead of guessed numbers.
Run: .venv/Scripts/python scripts/ocr_layout_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrtrans.ocr import RapidOcrEngine, group_blocks  # noqa: E402

W, H = 977, 193
BOLD = "C:/Windows/Fonts/segoeuib.ttf"
SERIF = "C:/Windows/Fonts/georgiab.ttf"
BG = (44, 35, 30)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def centered(draw, text, font, y, fill=(240, 240, 240)) -> None:
    width = draw.textlength(text, font=font)
    draw.text(((W - width) / 2, y), text, font=font, fill=fill)


def left(draw, text, font, x, y, fill=(240, 240, 240)) -> None:
    draw.text((x, y), text, font=font, fill=fill)


def case_title_dialogue(gap: int, name_px: int = 40, body_px: int = 30):
    img, draw = canvas()
    centered(draw, "Beleaguered Celestial", ImageFont.truetype(SERIF, name_px), 28, (240, 220, 170))
    centered(
        draw,
        "Oh, what am I supposed to do now...",
        ImageFont.truetype(BOLD, body_px),
        28 + name_px + gap,
    )
    return img


def case_wrapped(lines: int):
    img, draw = canvas()
    font = ImageFont.truetype(BOLD, 30)
    body = [
        "Not what you would expect from a Sentinel, is it?",
        "But that's exactly why, through all this,",
        "we haven't lost ourselves.",
    ][:lines]
    for index, text in enumerate(body):
        centered(draw, text, font, 30 + index * 44)
    return img


def case_name_two_lines():
    img, draw = canvas()
    centered(draw, "Beleaguered Celestial", ImageFont.truetype(SERIF, 40), 10, (240, 220, 170))
    centered(draw, "Oh, what am I supposed to do now...", ImageFont.truetype(BOLD, 30), 66)
    centered(draw, "Nothing left to say.", ImageFont.truetype(BOLD, 30), 106)
    return img


def case_same_size_name():
    img, draw = canvas()
    centered(draw, "Beleaguered Celestial", ImageFont.truetype(BOLD, 30), 50)
    centered(draw, "Oh, what am I supposed to do now...", ImageFont.truetype(BOLD, 30), 100)
    return img


def case_two_columns():
    img, draw = canvas()
    font = ImageFont.truetype(BOLD, 24)
    left(draw, "Quest: find the key", font, 40, 40)
    left(draw, "Reward: 300 gold", font, 40, 78)
    left(draw, "HP 100", font, 700, 40)
    left(draw, "Ammo 30", font, 700, 78)
    return img


def case_caption_with_prompt():
    img, draw = canvas()
    centered(draw, "Beleaguered Celestial", ImageFont.truetype(SERIF, 40), 20, (240, 220, 170))
    centered(draw, "Oh, what am I supposed to do now...", ImageFont.truetype(BOLD, 30), 80)
    left(draw, "F", ImageFont.truetype(BOLD, 34), 900, 60)
    return img


def case_menu():
    img, draw = canvas()
    font = ImageFont.truetype(BOLD, 24)
    for index, item in enumerate(("Ask about the gate", "Leave the village", "Say nothing")):
        left(draw, item, font, 60, 20 + index * 38)
    return img


def case_single_line():
    img, draw = canvas()
    centered(draw, "The end.", ImageFont.truetype(BOLD, 30), 70)
    return img


def case_quoted_dialogue():
    img, draw = canvas()
    centered(draw, "Beleaguered Celestial", ImageFont.truetype(SERIF, 40), 24, (240, 220, 170))
    centered(
        draw,
        "\u201cOh, what am I supposed to do now...\u201d",
        ImageFont.truetype(BOLD, 30),
        82,
    )
    return img


def case_long_name():
    img, draw = canvas()
    centered(draw, "High Priestess of the Sun", ImageFont.truetype(SERIF, 34), 30, (240, 220, 170))
    centered(draw, "The gate opens at dawn.", ImageFont.truetype(BOLD, 30), 84)
    return img


CASES = {
    "A name+dialogue gap 30": lambda: case_title_dialogue(30),
    "B name+dialogue gap 12": lambda: case_title_dialogue(12),
    "C name+dialogue gap 0": lambda: case_title_dialogue(0),
    "D name 34px dialogue 30px, gap 10": lambda: case_title_dialogue(10, name_px=34, body_px=30),
    "E wrapped 2 lines": lambda: case_wrapped(2),
    "F wrapped 3 lines": lambda: case_wrapped(3),
    "G name + 2 dialogue lines": case_name_two_lines,
    "H same-size name": case_same_size_name,
    "I two columns": case_two_columns,
    "J caption + F prompt": case_caption_with_prompt,
    "K menu list": case_menu,
    "L single line": case_single_line,
    "M quoted dialogue": case_quoted_dialogue,
    "N five-word name": case_long_name,
}


def main() -> None:
    engine = RapidOcrEngine(use_cuda=True)
    for name, build in CASES.items():
        img = build()
        frame = np.array(img)[:, :, ::-1].copy()
        boxes = sorted(engine.read(frame), key=lambda b: (b.rect.y, b.rect.x))
        print(f"\n=== {name} ===")
        for box in boxes:
            print(
                f"  Box(Rect({box.rect.x}, {box.rect.y}, {box.rect.w}, {box.rect.h}), "
                f"{box.text!r}, {box.confidence:.3f})"
            )
        blocks = group_blocks(boxes)
        print(f"  -> {len(blocks)} blocks")
        for block in blocks:
            print(f"     {block.rect} {block.text!r}")


if __name__ == "__main__":
    main()
