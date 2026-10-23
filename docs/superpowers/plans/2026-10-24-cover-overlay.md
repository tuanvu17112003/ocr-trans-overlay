# Cover-Mode Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The overlay covers the whole OCR block with an opaque background and draws Vietnamese inside it, never showing English or a stale translation.

**Architecture:** `OverlayWindow` keeps two rects per block: `cover` (the OCR block, always painted) and `rect` (the fitted text box, centered inside `cover`). `OverlayState` drops a block from render until a translation exists for its current source text. `group_blocks` stops merging same-row labels and keeps paragraphs up to 6 lines.

**Tech Stack:** Python 3.13, PyQt6, RapidOCR, NLLB, pytest.

**Spec:** `docs/superpowers/specs/2026-10-24-cover-overlay-design.md`

## Global Constraints

- Windows 11, Python 3.13, PyQt6, no new dependencies.
- All coordinates are physical pixels; paint maps through `devicePixelRatioF()`.
- Background must cover exactly the OCR block; text never leaves it.
- Painted text is always Vietnamese; English source text is never drawn.
- Code, tests and commits in English; chat in Vietnamese.
- Default `pytest` stays fast; GPU and real-game tests stay opt-in.

## Review Focus

1. A paragraph whose Vietnamese is much shorter than the English: the whole block is still painted, no English line remains below the text.
2. Two labels on the same row (`Vietnamese` `English`): two blocks, two translations, never one merged block.
3. A 5-8 line paragraph: one block, one translation (cap only stops runaway list merges).
4. Source text changes between OCR passes: the previous translation disappears on that same pass.
5. Translation still loading for a new block: nothing is painted there (no English inside a dark bar).

---

### Task 1: Cover the whole block and center the text

**Files:**
- Modify: `src/ocrtrans/overlay.py`
- Test: `tests/test_overlay_cover.py` (create)

**Interfaces:**
- Produces: `_Layout(item: RenderItem, cover: Rect, rect: Rect, font_px: int, lines: list[str])`; `cover` is the source block and the only rect painted with the background.
- Keeps: `OverlayWindow.show_items(items)`, `OverlayWindow._layouts`.

- [ ] **Step 1: Write the failing tests**

```python
import queue

from PyQt6.QtWidgets import QApplication

from ocrtrans.config import Config
from ocrtrans.geometry import Rect
from ocrtrans.overlay import OverlayWindow
from ocrtrans.pipeline import RenderItem


def make_item(text="Xin chào", rect=Rect(100, 100, 200, 80)):
    return RenderItem(
        block_id=1, text=text, rect=rect, source_text="Hello world", cached=False, line_px=20
    )


def make_overlay(qapp):
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), queue.Queue())
    overlay.show_items([make_item()])
    return overlay


def test_cover_is_the_source_block(qapp):
    overlay = make_overlay(qapp)
    layout = overlay._layouts[0]
    assert layout.cover == Rect(100, 100, 200, 80)
    assert layout.rect.x >= 100 and layout.rect.y >= 100
    assert layout.rect.x + layout.rect.w <= 300
    assert layout.rect.y + layout.rect.h <= 180
    overlay.close()


def test_text_is_centered_vertically_inside_the_cover(qapp):
    overlay = make_overlay(qapp)
    layout = overlay._layouts[0]
    top_gap = layout.rect.y - 100
    bottom_gap = 180 - (layout.rect.y + layout.rect.h)
    assert abs(top_gap - bottom_gap) <= 2
    overlay.close()


def test_background_fills_the_cover_with_opaque_color(qapp):
    overlay = make_overlay(qapp)
    image = overlay.grab().toImage()
    color = image.pixelColor(102, 102)
    assert color.alpha() == 255
    assert color.name() == Config().bg_color.lower()
    overlay.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_overlay_cover.py -q`
Expected: FAIL (`_Layout` has no `cover`; background is the fitted rect).

- [ ] **Step 3: Implement**

Add `cover: Rect` to `_Layout`. In `_fit`, pass an inner rect (`item.rect` inset by 2 px on the sides) to `fit_text` and center the result vertically inside `item.rect`:

```python
inner = Rect(item.rect.x + 2, item.rect.y, max(1, item.rect.w - 4), item.rect.h)
fit = fit_text(item.text, inner, self.screen_rect, self.measure, line_px=item.line_px,
               min_font_px=self.config.min_font_px)
y = inner.y + max(0, (inner.h - fit.rect.h) // 2)
return _Layout(item=item, cover=item.rect, rect=Rect(inner.x, y, fit.rect.w, fit.rect.h),
               font_px=fit.font_px, lines=fit.lines)
```

Drop the obstacles/second pass from `_recompute` (text can no longer leave its own cover). In `paintEvent`, fill `layout.cover`, clip to `layout.cover`, then draw `layout.lines` from `layout.rect.y` with `line_h`, honoring `overlay_opacity` only for the background.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_overlay_cover.py tests/test_e2e_smoke.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/overlay.py tests/test_overlay_cover.py
git commit -m "fix: cover the whole OCR block and center the translation inside it"
```

---

### Task 2: Never paint a translation for the wrong text

**Files:**
- Modify: `src/ocrtrans/lifecycle.py`
- Test: `tests/test_lifecycle.py`

**Interfaces:**
- Produces: `OverlayState` renders a block only when `block.translated` is set for the current `block.source`; changing the source clears `translated` immediately.
- Keeps: `OverlayState.on_event`, `RenderItem` fields.

- [ ] **Step 1: Write the failing tests**

```python
def test_old_translation_disappears_when_source_changes():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HELLO")]), 0.0)
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHAO", source="HELLO")]), 0.1)
    assert len(s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HELLO")]), 0.2)) == 1
    assert s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HELLO THERE")]), 0.3) == []


def test_block_stays_hidden_until_its_translation_arrives():
    s = OverlayState()
    assert s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HELLO")]), 0.0) == []
    rendered = s.on_event(
        ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHAO", source="HELLO")]), 0.1
    )
    assert [r.text for r in rendered] == ["XIN CHAO"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_lifecycle.py -q`
Expected: FAIL (the first pass renders `"HELLO"`, the changed source keeps `"XIN CHAO"`).

- [ ] **Step 3: Implement**

In `_on_ocr`, when a shown block's source text changes, clear its translation (`block.translated = ""`). In `_render`, `continue` when `block.translated` is empty and pass `text=block.translated` instead of `block.translated or block.source`. Compare normalized text (case and spaces folded) both in `_on_ocr` and in `_on_translation` so OCR case/spacing jitter does not flicker the block.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_lifecycle.py tests/test_overlay_cover.py tests/test_e2e_smoke.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/lifecycle.py tests/test_lifecycle.py
git commit -m "fix: hide a block until its translation matches the current source"
```

---

### Task 3: Stop merging same-row labels and keep longer paragraphs

**Files:**
- Modify: `src/ocrtrans/ocr.py` (`group_blocks`)
- Test: `tests/test_ocr.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `group_blocks(boxes, max_gap_ratio=0.6, min_confidence=0.6, max_lines=6)`; a line joins the previous one only when `gap >= -0.2 * line_height`.
- Keeps: `Block.rect`, `Block.text`, `Block.line_px`.

- [ ] **Step 1: Write the failing tests**

```python
def test_same_row_labels_are_not_merged():
    boxes = [
        Box(Rect(10, 100, 120, 22), "Vietnamese", 0.9),
        Box(Rect(150, 100, 120, 22), "English", 0.9),
    ]
    assert len(group_blocks(boxes)) == 2


def test_five_line_paragraph_stays_one_block():
    boxes = [Box(Rect(10, 10 + index * 18, 300, 16), f"line {index}", 0.9) for index in range(5)]
    blocks = group_blocks(boxes)
    assert len(blocks) == 1
    assert len(blocks[0].lines) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr.py -q`
Expected: FAIL (labels merge; the fifth line is split off).

- [ ] **Step 3: Implement**

Replace the gap window `-line_h <= gap <= max_gap_ratio * line_h` with `-0.2 * line_h <= gap <= max_gap_ratio * line_h` and change `max_lines` to 6.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_ocr.py tests/test_pipeline.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/ocr.py tests/test_ocr.py
git commit -m "fix: keep same-row labels apart and group paragraphs up to six lines"
```

---

### Task 4: Verify on the real game and prove the cover visually

**Files:**
- Create: `.superpowers/cover_proof.py` (not committed)
- Modify: `README.md` (cover-mode rule)

- [ ] **Step 1: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass, machine/real tests skipped.

- [ ] **Step 2: Run GPU and real-game tests**

Run: `.venv/Scripts/python -m pytest -m gpu -q`
Run with Gunfire Reborn open in borderless mode: `.venv/Scripts/python -m pytest -m real tests/real --run-real -q -s`
Expected: PASS; record OCR/translation timings.

- [ ] **Step 3: Render a proof image**

`.superpowers/cover_proof.py` draws an English paragraph on a dark canvas, runs real OCR, `group_blocks`, NLLB and `OverlayState`, renders the overlay offscreen with `OverlayWindow`, composites it over the source, and writes `.superpowers/cover_proof.png`. It asserts that every pixel inside a block is either the background color or a glyph of the Vietnamese text (no source pixel survives).

- [ ] **Step 4: Smoke the app**

Run: `.venv/Scripts/python -m ocrtrans.app --smoke`
Expected: exit 0, `ocr_passes >= 1`.

- [ ] **Step 5: Commit docs**

```bash
git add README.md
git commit -m "docs: describe cover-mode overlay behaviour"
```
