"""Candidate gate: artwork must not reach the translator.

Thresholds here are pinned to measurements, not to guesses. The number in each
comment is what the gate was tuned against on this machine.
"""

import numpy as np

from ocrtrans.candidates import (
    GateStats,
    filter_candidates,
    stroke_ratio,
    text_likeness,
)
from ocrtrans.geometry import Box, Rect


def box(text, x=0, y=0, w=200, h=40, confidence=0.9):
    return Box(Rect(x, y, x * 0 + w, h) if False else Rect(x, y, w, h), text, confidence)


def test_a_normal_caption_passes():
    assert text_likeness("Open the gate, traveler.", Rect(0, 0, 420, 42)).keep


def test_a_tall_artwork_box_is_rejected_by_aspect():
    # 60x178 (aspect 0.34) is what a texture patch looks like; one Latin line
    # of text cannot have that shape.
    assert not text_likeness("x.O^", Rect(0, 0, 60, 178)).keep


def test_density_rejects_a_large_blob_that_reads_as_letters():
    # The historic "TANOV" detector artefact: a 1059x317 blob reading five chars.
    assert not text_likeness("TANOV", Rect(0, 0, 1059, 317)).keep


def test_symbol_runs_are_rejected_by_content():
    assert not text_likeness("x.O^ O^III", Rect(0, 0, 120, 40)).keep
    assert not text_likeness("", Rect(0, 0, 120, 40)).keep
    assert not text_likeness("## x.O", Rect(0, 0, 120, 40)).keep


def test_short_hud_number_is_rejected_by_the_subtitle_profile():
    # "100" has no letters, so the subtitle profile drops it; the hud profile
    # is the opt-in that keeps it.
    assert not text_likeness("100", Rect(0, 0, 60, 22)).keep
    assert text_likeness("100", Rect(0, 0, 60, 22), "hud").keep


def test_real_corpus_lines_all_pass_the_content_check():
    """Every line of the 80-line benchmark corpus must survive the gate."""
    import json
    from pathlib import Path

    corpus = Path(__file__).resolve().parents[1] / "src/ocrtrans/bench/corpus/quality.jsonl"
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    rejected = []
    for row in rows:
        # A typical rendered line: 30 px tall, width proportional to the text.
        rect = Rect(0, 0, max(120, len(row["text"]) * 12), 30)
        verdict = text_likeness(row["text"], rect, "subtitle")
        if not verdict.keep:
            rejected.append((row["text"], verdict.reason))
    assert rejected == [], f"the gate dropped real captions: {rejected[:5]}"


def test_confidence_gate_uses_the_profile_threshold():
    stats = GateStats()
    kept = filter_candidates(
        [box("Open the gate", confidence=0.55)],
        profile="subtitle",
        stats=stats,
    )
    assert kept == []
    assert stats.rejected == 1


def test_a_caption_filling_most_of_a_short_strip_is_kept():
    # Regression: a 40 px font in a 200 px subtitle strip is 55 px tall, i.e.
    # 27 % of the region. A 25 % height cap dropped every real caption in a
    # strip-shaped region.
    kept = filter_candidates(
        [box("Open the gate, traveler.", w=424, h=55)],
        profile="subtitle",
        region=Rect(0, 0, 900, 200),
    )
    assert [b.text for b in kept] == ["Open the gate, traveler."]


def test_a_blob_taller_than_half_the_region_is_rejected():
    kept = filter_candidates(
        [box("Some words here now", w=400, h=120)],
        profile="subtitle",
        region=Rect(0, 0, 2000, 200),
    )
    assert kept == []


def test_stroke_ratio_is_diagnostic_only_and_the_bands_overlap():
    """Records the measurement that stopped the gate from using stroke width."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 40)
    image = Image.new("L", (420, 60), 0)
    ImageDraw.Draw(image).text((10, 8), "Open the gate", font=font, fill=255)
    text_value = stroke_ratio(np.array(image))

    icon = Image.new("L", (120, 178), 40)
    draw = ImageDraw.Draw(icon)
    for row in range(2):
        for col in range(2):
            x, y = 10 + col * 55, 10 + row * 85
            draw.ellipse((x, y, x + 40, y + 40), fill=200)
            draw.rectangle((x + 8, y + 12, x + 30, y + 28), fill=60)
    icon_value = stroke_ratio(np.array(icon))

    # Both are inside 0.02-0.35, which is exactly why the band cannot be a gate.
    assert 0.02 <= text_value <= 0.35
    assert 0.02 <= icon_value <= 0.35


def test_the_gate_drops_an_icon_grid_but_keeps_a_caption():
    boxes = [
        box("x.O^", x=0, y=0, w=60, h=178),
        box("O^III", x=80, y=0, w=52, h=178),
        box("Open the gate, traveler.", x=0, y=200, w=420, h=42),
    ]
    stats = GateStats()
    kept = filter_candidates(boxes, profile="subtitle", stats=stats)
    assert [b.text for b in kept] == ["Open the gate, traveler."]
    assert stats.rejected == 2
    assert stats.reasons