# Text-first overlay pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace pixel-diff OCR scheduling and miss-count overlay state with a text-first scan loop that works on moving backgrounds and never shows a stale translation.

**Architecture:** A capture thread grabs at `fps`, and a `ScanBudget` decides when a full-region OCR pass may start (one pass in flight, interval measured from the previous pass end). `BlockMemory` matches blocks by rect, requires exact text for stability, and keeps a block for a short TTL. `OverlayState` is driven by complete snapshots ordered by `scan_id` and paints a block only when its translation was requested for exactly the source text on screen.

**Tech Stack:** Python 3.13, PyQt6, RapidOCR (ONNX Runtime), numpy, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-10-24-text-first-overlay-design.md`

## Global Constraints

- Windows 10/11 only. No process injection, no memory reads, no global keyboard hooks (`RegisterHotKey` only).
- The overlay stays click-through and excluded from capture (`WDA_EXCLUDEFROMCAPTURE`).
- Offline by default. Online engines require `Config.online_consent`; do not change that gate.
- No new required dependency in this plan.
- Exact normalised text decides translation validity. No similarity/fuzzy rule anywhere in the block-memory or paint path.
- At most one OCR pass in flight. The capture thread never replaces a pending frame.
- Code, comments, docs and commits in English. User-visible UI strings follow the existing control window.
- Test commands: `.venv/Scripts/python -m pytest -q`, `.venv/Scripts/python -m pytest -m gpu -q`, `.venv/Scripts/python -m pytest tests/real --run-real -q` (skips when the game is closed).

## Review Focus

These are the failures the spec implies but no single task's happy path exercises. Each has a named test in the task that owns the code.

1. **Frame size differs from the region** (HDR/DPI scaling) -> boxes must still map to screen coordinates. Task 4: `test_emitted_rects_are_screen_coordinates`.
2. **One false-negative OCR pass on a good screen** -> the subtitle must not blink off. Task 1: `test_block_survives_one_empty_pass_within_ttl`.
3. **Digit-only UI change** ("100 HP" -> "10 HP") -> the old translation must never paint. Task 3: `test_digit_change_never_keeps_the_old_translation`; Task 7 repeats it end to end.
4. **Translation arrives after the subtitle changed** -> drop it, do not paint it over the new text. Task 3: `test_translation_for_old_source_is_dropped`.
5. **Wide region, sparsely placed small text** (found nothing at the 1280 cap) -> the bounded tiled rescue must find it. Task 5: `test_tiled_rescue_finds_text_a_single_pass_missed`.

---

### Task 1: BlockMemory

**Files:**
- Create: `src/ocrtrans/blocks.py`
- Test: `tests/test_blocks.py`

**Interfaces:**
- Consumes: `geometry.Rect`, `geometry.Box`.
- Produces: `blocks.BlockState` (`PENDING`, `STABLE`), `blocks.TrackBlock(id, rect, text, state, first_seen, last_seen, line_px)`, `blocks.match_score(a: Rect, b: Rect) -> float`, `blocks.BlockMemory(profile="safe" | "fast")` with `set_profile(profile: str) -> None`, `reset() -> None`, `observe(boxes: list[Box], now: float, ttl_s: float) -> list[TrackBlock]`, and `blocks.STABILITY_WINDOWS_S = {"safe": 0.300, "fast": 0.150}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_blocks.py
from ocrtrans.blocks import BlockMemory, BlockState, match_score
from ocrtrans.geometry import Box, Rect


def box(text, x=0, y=0, w=100, h=20, line_px=18):
    return Box(Rect(x, y, w, h), text, 0.9, line_px=line_px)


def test_same_text_becomes_stable_after_the_safe_window():
    m = BlockMemory("safe")
    assert m.observe([box("HELLO")], 0.0, 0.4)[0].state is BlockState.PENDING
    assert m.observe([box("HELLO")], 0.20, 0.4)[0].state is BlockState.PENDING
    assert m.observe([box("HELLO")], 0.31, 0.4)[0].state is BlockState.STABLE


def test_fast_profile_uses_a_shorter_window():
    m = BlockMemory("fast")
    m.observe([box("HELLO")], 0.0, 0.4)
    assert m.observe([box("HELLO")], 0.16, 0.4)[0].state is BlockState.STABLE


def test_one_digit_change_resets_stability():
    m = BlockMemory("fast")
    m.observe([box("You have 100 HP")], 0.0, 1.0)
    stable = m.observe([box("You have 100 HP")], 0.2, 1.0)[0]
    assert stable.state is BlockState.STABLE
    changed = m.observe([box("You have 10 HP")], 0.3, 1.0)[0]
    assert changed.text == "You have 10 HP"
    assert changed.state is BlockState.PENDING
    assert changed.first_seen == 0.3
    assert m.observe([box("You have 10 HP")], 0.4, 1.0)[0].state is BlockState.PENDING


def test_case_and_spacing_wobble_is_the_same_text():
    m = BlockMemory("fast")
    m.observe([box("HELLO WORLD")], 0.0, 1.0)
    block = m.observe([box("  hello   world ")], 0.2, 1.0)[0]
    assert block.state is BlockState.STABLE


def test_identity_survives_small_rect_drift():
    m = BlockMemory("fast")
    first = m.observe([box("HELLO", x=10, y=10)], 0.0, 1.0)[0]
    second = m.observe([box("HELLO", x=12, y=11)], 0.1, 1.0)[0]
    assert first.id == second.id


def test_two_boxes_never_match_the_same_block():
    m = BlockMemory("fast")
    m.observe([box("A", y=10), box("B", y=60)], 0.0, 1.0)
    blocks = m.observe([box("B", y=10), box("A", y=60)], 0.1, 1.0)
    by_rect = {b.rect.y: b.text for b in blocks}
    assert by_rect == {10: "B", 60: "A"}
    assert len({b.id for b in blocks}) == 2


def test_block_survives_one_empty_pass_within_ttl():
    m = BlockMemory("fast")
    m.observe([box("HELLO")], 0.0, 0.5)
    assert [b.text for b in m.observe([], 0.3, 0.5)] == ["HELLO"]
    assert m.observe([], 0.51, 0.5) == []


def test_a_moved_block_with_new_text_does_not_ghost_at_the_old_rect():
    m = BlockMemory("fast")
    m.observe([box("AAAA", x=0, w=100)], 0.0, 1.0)
    blocks = m.observe([box("BBBB", x=60, w=100)], 0.1, 1.0)
    assert [(b.rect.x, b.text) for b in blocks] == [(60, "BBBB")]


def test_match_score_prefers_overlap_and_containment():
    a = Rect(0, 0, 100, 20)
    assert match_score(a, Rect(0, 0, 100, 20)) == 1.0
    assert match_score(a, Rect(0, 0, 100, 24)) >= 0.6
    assert match_score(a, Rect(500, 500, 10, 10)) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_blocks.py -q`
Expected: `ModuleNotFoundError: No module named 'ocrtrans.blocks'`.

- [ ] **Step 3: Implement `src/ocrtrans/blocks.py`**

Copy `match_score` and `_norm` from `track.py` unchanged (`_norm` = `" ".join(text.casefold().split())`). `observe` logic:

- For each box, pick the unmatched block with the highest `match_score(block.rect, box.rect)` above `0.5`; otherwise create a block with a fresh id.
- If `_norm(box.text) != _norm(block.text)`: set `text = box.text`, `first_seen = now`, `state = PENDING`.
- If equal: keep the existing `text`, and set `state = STABLE` when `now - first_seen >= STABILITY_WINDOWS_S[profile]` (unknown profile -> safe).
- Always set `last_seen = now`, `rect = box.rect`, and `line_px = box.line_px or keep`.
- Drop unmatched blocks older than `ttl_s`; return the rest ordered by `id`.
- Also drop an unmatched block when its rect overlaps a **matched** block with IoU `>= 0.1`, even inside its TTL: a paragraph that moved or was re-read with new text must not ghost at the old rect (a stale cover on screen is the anti-goal).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_blocks.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/blocks.py tests/test_blocks.py
git commit -m "feat: add exact-text block memory"
```

---

### Task 2: ScanBudget and frame wake

**Files:**
- Create: `src/ocrtrans/scan.py`
- Test: `tests/test_scan.py`

**Interfaces:**
- Consumes: `numpy`, `cv2`.
- Produces: `scan.ScanBudget(fps: int = 15)` with `interval(last_ocr_ms: float, empty_count: int = 0, wake: bool = False) -> float`, `set_fps(fps: int) -> None`, `ttl(interval_s: float) -> float`, `frame_changed(prev: np.ndarray | None, frame: np.ndarray, diff_luma: int = 12, threshold: float = 0.01) -> tuple[bool, np.ndarray]` (module-level function), and the constants `MIN_INTERVAL_S=0.25`, `OCR_HEADROOM=1.5`, `EMPTY_BACKOFF_S=1.0`, `EMPTY_PASSES=3`, `TTL_MIN_S=0.4`, `TTL_MAX_S=1.0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scan.py
import numpy as np
import pytest

from ocrtrans.scan import ScanBudget, frame_changed


@pytest.mark.parametrize(
    "fps,last_ms,empty,wake,expected",
    [
        (15, 300, 0, False, 0.450),
        (15, 300, 0, True, 0.450),
        (15, 300, 3, False, 1.000),
        (15, 300, 3, True, 0.450),
        (15, 852, 3, False, 1.278),
        (15, 852, 3, True, 1.278),
        (15, 20, 0, False, 0.250),
        (15, 20, 5, False, 1.000),
    ],
)
def test_scan_budget_matches_the_pinned_table(fps, last_ms, empty, wake, expected):
    assert ScanBudget(fps).interval(last_ms, empty, wake) == pytest.approx(expected)


def test_ttl_is_two_intervals_clamped_between_400ms_and_1s():
    b = ScanBudget(15)
    assert b.ttl(0.25) == pytest.approx(0.5)
    assert b.ttl(0.1) == pytest.approx(0.4)
    assert b.ttl(0.852) == pytest.approx(1.0)


def test_frame_changed_flags_noise_but_not_a_repeat():
    frame = np.full((100, 320, 3), 30, np.uint8)
    changed, gray = frame_changed(None, frame)
    assert changed is False
    assert gray.shape[0] == 100
    assert frame_changed(gray, frame)[0] is False
    noise = np.random.default_rng(0).integers(0, 255, frame.shape, dtype=np.uint8)
    assert frame_changed(gray, noise)[0] is True


def test_frame_changed_survives_a_region_size_change():
    _, gray = frame_changed(None, np.full((100, 320, 3), 30, np.uint8))
    changed, new_gray = frame_changed(gray, np.full((100, 400, 3), 30, np.uint8))
    assert changed is False
    assert new_gray.shape != gray.shape
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_scan.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/ocrtrans/scan.py`**

`interval` is exactly the spec 4.5 pseudocode (`base = max(1/fps, MIN_INTERVAL_S, OCR_HEADROOM*last_ocr_ms/1000)`, empty backoff only when `not wake`). `set_fps` assigns `self.fps` so `Pipeline.set_fps` can reach it. `ttl` is `min(max(2*interval, TTL_MIN_S), TTL_MAX_S)`. `frame_changed` reproduces the old `_gray` step: grayscale if needed, resize to 320 px wide, `GaussianBlur(5,5)`, `absdiff > diff_luma`, then `changed = float(diff.mean()) > threshold` (a Python `bool`, not `np.bool_`); return `(changed, gray)`. If `prev is None` **or `prev.shape != gray.shape`** (region change, backend switch), return `(False, gray)` so `cv2.absdiff` can never raise in the capture thread.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_scan.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/scan.py tests/test_scan.py
git commit -m "feat: add scan budget with a pinned interval table"
```

---

### Task 3: Event contract and overlay state machine

**Files:**
- Modify: `src/ocrtrans/pipeline.py` (event dataclasses, `_scan_id`, drop `SKIP` emission and `observed_block_ids`)
- Rewrite: `src/ocrtrans/lifecycle.py`
- Modify: `src/ocrtrans/overlay.py:136-147` (`_drain` calls `tick`) and add a `shown` property
- Rewrite: `tests/test_lifecycle.py`
- Modify: `tests/test_pipeline.py` (delete the two tests that assert removed fields)

**Interfaces:**
- Consumes: `blocks` is not used here; `capture.Health`; `pipeline.PipelineEvent`.
- Produces: `PipelineEvent.scan_id: int = 0`, `PipelineEvent.scan_interval_s: float = 0.0`, `PipelineEvent.scan_ms: float = 0.0`; `EventKind` without `SKIP`; `OverlayState(unhealthy_hold_s=1.0, watchdog_s=1.5)` with `on_event(event, now) -> list[RenderItem]`, `tick(now) -> list[RenderItem] | None` (`None` = unchanged), `state(block_id) -> str` (`hidden` | `pending` | `shown`), `hidden_stale: int`, `clear()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lifecycle.py
from ocrtrans.capture import Health
from ocrtrans.geometry import Rect
from ocrtrans.lifecycle import OverlayState
from ocrtrans.pipeline import BlockView, EventKind, PipelineEvent, RenderItem


def item(block_id, text, source):
    return RenderItem(block_id, text, Rect(0, 0, 10, 10), source, False, 0)


def block(block_id, text, state="stable"):
    return BlockView(block_id, Rect(0, 0, 10, 10), text, state, 18)


def ev(kind, epoch=1, scan_id=1, **kw):
    return PipelineEvent(kind=kind, epoch=epoch, at=0.0, scan_id=scan_id, **kw)


def shown_state(text="HELLO", translation="XIN CHÀO"):
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, text)]), 0.0)
    s.on_event(ev(EventKind.TRANSLATION_RESULT, scan_id=1, items=[item(1, translation, text)]), 0.1)
    return s


def test_a_block_that_left_the_snapshot_disappears_at_once():
    s = shown_state()
    assert [i.text for i in s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[]), 0.2)] == []


def test_digit_change_never_keeps_the_old_translation():
    s = shown_state("You have 100 HP", "Bạn có 100 HP")
    assert s.on_event(
        ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "You have 10 HP")]), 0.2
    ) == []


def test_translation_for_old_source_is_dropped():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "NEW TEXT")]), 0.0)
    out = s.on_event(
        ev(EventKind.TRANSLATION_RESULT, scan_id=1, items=[item(1, "CŨ", "OLD TEXT")]), 0.1
    )
    assert out == []


def test_stale_snapshot_is_ignored():
    s = shown_state("NEW", "MỚI")
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "NEW")]), 0.2)
    out = s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, "OLD")]), 0.3)
    assert [i.text for i in out] == ["MỚI"]


def test_fallback_capture_keeps_painting():
    s = shown_state()
    assert [i.text for i in s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.WGC_FALLBACK), 0.5)] == ["XIN CHÀO"]


def test_black_capture_hides_after_one_second():
    s = shown_state()
    s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.BLACK), 0.5)
    assert s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.BLACK), 1.6) == []


def test_health_recovery_alone_does_not_repaint():
    s = shown_state()
    s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.BLACK), 0.5)
    s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.BLACK), 1.6)
    assert s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.OK), 1.7) == []


def test_watchdog_suppresses_then_repaints_on_a_fresh_snapshot_without_new_translation():
    s = OverlayState(watchdog_s=1.5)
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, "HELLO")]), 0.0)
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHÀO", "HELLO")]), 0.1)
    assert s.tick(1.6) == []
    assert s.tick(1.7) is None
    out = s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "HELLO")]), 1.8)
    assert [i.text for i in out] == ["XIN CHÀO"]


def test_watchdog_scales_with_a_slow_pass_cycle():
    s = OverlayState()
    s.on_event(
        ev(
            EventKind.OCR_RESULT,
            scan_id=1,
            scan_interval_s=1.278,
            scan_ms=852,
            blocks=[block(1, "HELLO")],
        ),
        0.0,
    )
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHÀO", "HELLO")]), 0.1)
    assert s.tick(2.0) is None          # watchdog = 1.278 + 0.852 + 0.35 = 2.48 s
    assert s.tick(2.5) == []


def test_stale_cover_does_not_return_after_a_capture_gap():
    s = OverlayState(watchdog_s=1.5)
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, "A")]), 0.0)
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "A-VI", "A")]), 0.1)
    assert s.tick(1.6) == []
    s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.OK), 1.7)
    assert s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "B")]), 1.8) == []
    assert s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "A-VI", "A")]), 1.9) == []


def test_reset_clears_and_older_epoch_is_ignored():
    s = shown_state("NEW", "MỚI")
    assert s.on_event(ev(EventKind.RESET, epoch=2), 0.3) == []
    assert s.state(1) == "hidden"
    out = s.on_event(ev(EventKind.OCR_RESULT, epoch=1, scan_id=1, blocks=[block(1, "OLD")]), 0.4)
    assert out == []


def test_reset_accepts_the_first_snapshot_of_the_new_epoch():
    s = OverlayState()
    for scan in range(1, 5):
        s.on_event(ev(EventKind.OCR_RESULT, scan_id=scan, blocks=[block(1, "HELLO")]), 0.1 * scan)
    s.on_event(ev(EventKind.RESET, epoch=2), 0.5)
    s.on_event(ev(EventKind.OCR_RESULT, epoch=2, scan_id=1, blocks=[block(1, "HELLO")]), 0.6)
    out = s.on_event(
        ev(EventKind.TRANSLATION_RESULT, epoch=2, scan_id=1, items=[item(1, "XIN CHÀO", "HELLO")]),
        0.7,
    )
    assert [i.text for i in out] == ["XIN CHÀO"]


def test_a_removed_block_is_counted_as_hidden_stale():
    s = shown_state()
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[]), 0.2)
    assert s.hidden_stale == 1


def test_render_keeps_source_line_height():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, "HELLO")]), 0.0)
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHÀO", "HELLO")]), 0.1)
    out = s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "HELLO")]), 0.2)
    assert out[0].line_px == 18


def test_pending_block_is_not_rendered():
    s = OverlayState()
    out = s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HEL", state="pending")]), 0.0)
    assert out == []
```

Also delete `test_crop_scan_reports_only_observed_ids` and `test_skip_is_reported_as_skip_not_empty_ocr` from `tests/test_pipeline.py` (they assert the removed fields/event).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_lifecycle.py -q`
Expected: failures on `scan_id` (unexpected keyword) and on the new behavior.

- [ ] **Step 3: Implement**

`pipeline.py`: add `scan_id: int = 0`, `scan_interval_s: float = 0.0`, `scan_ms: float = 0.0` to `PipelineEvent`; remove `observed_block_ids`; remove `SKIP` from `EventKind`; add `self._scan_id = 0`, reset it in `set_regions`/`adopt_engine`/`set_enabled`. **Bridge until Task 4:** on every capture tick emit `OCR_RESULT` with `scan_id` incremented and the tracker's complete live list (`self.tracker.blocks()`), converting each view with `rect=_offset(block.rect, origin)` where `origin` is the task's origin on the pass branch and `self._region()` on the `decision.skip` branch; pass `scan_interval_s=0.0, scan_ms=0.0` (the watchdog stays 1.5 s until Task 4 fills them). The new overlay removes blocks by absence, so the old crop-scoped `seen` list and the SKIP event cannot survive this task; Task 4 deletes the bridge. `FakeTracker` in `tests/test_pipeline.py` becomes stateful: `observe` stores the returned list, `blocks()` returns it (so the complete snapshot matches the pass result and `test_empty_ocr_result_carries_no_blocks` still holds).

`lifecycle.py`: exact-text rule (normalise with casefold + whitespace collapse); on `OCR_RESULT` ignore `scan_id <= last_scan_id` by returning `self._render(now)` unchanged (no state mutation, never `[]`), update/create blocks, clear `translated` when the source text differs, remove ids missing from the snapshot (increment `self.hidden_stale` per removed block), update `last_scan_id`/`last_ocr_at`, then lift suppression only when `scan_id > suppressed_at_scan`; on `TRANSLATION_RESULT` apply only when the normalised source matches; on `CAPTURE_HEALTH` treat `OK`/`None`/`WGC_FALLBACK` as no hold, otherwise suppress after `unhealthy_hold_s`; on every applied snapshot set `watchdog_s = max(1.5, event.scan_interval_s + event.scan_ms / 1000 + 0.35)` so a slow machine (852 ms pass, 1.278 s interval) cannot trip the watchdog between two normal snapshots; `tick` suppresses when `now - last_ocr_at >= watchdog_s` and returns `None` when nothing changed. **Evaluate the watchdog at the end of `on_event`** (after the event has been applied) and at the start of `tick`, so a snapshot that arrives after a gap first refreshes `last_ocr_at`. On `RESET`: clear blocks, `last_scan_id = 0`, `suppressed = False`, `suppressed_at_scan = 0`, `last_ocr_at = None`, `hidden_stale = 0`. `_render` paints only when `translated` is non-empty and `translated_for` matches `source`.

`overlay.py`: in `_drain`, after draining the queue, call `items = self.state.tick(now)`; if not `None`, assign and repaint. Add `@property def shown(self) -> int: return len(self._items)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_lifecycle.py tests/test_pipeline.py -q`
Expected: lifecycle passes; the remaining pipeline tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/pipeline.py src/ocrtrans/lifecycle.py src/ocrtrans/overlay.py tests/test_lifecycle.py tests/test_pipeline.py
git commit -m "feat: exact-text overlay state with scan ordering and render suppression"
```

---

### Task 4: Text-first scan loop

**Files:**
- Modify: `src/ocrtrans/pipeline.py` (constructor and capture/OCR loops)
- Modify: `src/ocrtrans/app.py:148-195, 222-227` (constructor, `set_profile`, status)
- Delete: `src/ocrtrans/track.py`, `tests/test_track.py`
- Rewrite: `tests/test_pipeline.py` (fakes and the tests that depended on the tracker)
- Modify: `tests/test_e2e_smoke.py`, `tests/test_empty_screen.py`, `tests/real/test_game.py`
- Modify: `src/ocrtrans/bench/__main__.py` (drop the `TemporalTracker` import and the positional tracker argument)

**Interfaces:**
- Consumes: `blocks.BlockMemory`, `scan.ScanBudget`, `scan.frame_changed`, `ocr.group_blocks`.
- Produces: `Pipeline(capture, ocr, service, emit, *, memory: BlockMemory | None = None, budget: ScanBudget | None = None, gate=None, fps=15, clock=time.monotonic)`; `Pipeline.set_profile(profile: str) -> None`; `Pipeline.set_fps(fps: int) -> None`; stats keys `scan_rate`, `tiled`, `last_scan_ms` (the last two filled in Task 5/6).

- [ ] **Step 1: Write the failing tests**

Rewrite `tests/test_pipeline.py` around the new constructor. Keep the existing engine/lifecycle/stats tests, and add (helpers: `NoiseCapture`, `CountingCapture`, `BlockingOcr`, `SlowOcr`):

```python
def test_moving_background_does_not_increase_the_scan_rate():
    p = pipeline_with(capture=NoiseCapture(), fps=60)
    p.start()
    time.sleep(1.2)
    passes = p.stats()["ocr_passes"]
    p.stop()
    assert 2 <= passes <= 6          # floor is 250 ms, not one pass per frame


def test_capture_does_not_grab_while_ocr_is_running():
    cap, ocr = CountingCapture(), BlockingOcr()
    p = pipeline_with(capture=cap, ocr=ocr, fps=60)
    p.start()
    assert ocr.started.wait(2.0)
    time.sleep(0.05)                 # let any in-flight grab land
    grabs = cap.grabs
    time.sleep(0.15)
    assert cap.grabs == grabs
    ocr.release.set()
    p.stop()


def test_scan_interval_respects_measured_ocr_time():
    p = pipeline_with(ocr=SlowOcr(0.30), fps=60)
    p.start()
    time.sleep(1.4)
    passes = p.stats()["ocr_passes"]
    p.stop()
    assert passes <= 3               # 1.5 * 300 ms floor


def test_empty_results_back_off_to_one_pass_per_second():
    p = pipeline_with(ocr=FakeOcr(boxes=[]), fps=60)
    p.start()
    time.sleep(2.2)
    passes = p.stats()["ocr_passes"]
    p.stop()
    assert 2 <= passes <= 5


def test_fallback_capture_still_runs_ocr():
    p = pipeline_with(
        capture=FallbackCapture(),
        ocr=FakeOcr(boxes=[Box(Rect(0, 0, 50, 12), "XXX", 0.9)]),
        fps=60,
    )
    p.start()
    ev = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.blocks)
    p.stop()
    assert ev is not None and ev.blocks[0].source_text == "XXX"


def test_scan_ids_increase_and_reset_with_the_epoch():
    p = pipeline_with(fps=60)
    p.start()
    p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT)
    first = [e.scan_id for e in p.events() if e.kind == EventKind.OCR_RESULT]
    before = len(p.events())
    p.set_regions([Rect(0, 0, 200, 100)])
    time.sleep(0.6)
    after = p.events()[before:]
    p.stop()
    assert first == sorted(set(first)) and first[0] == 1
    assert any(e.kind is EventKind.RESET for e in after)
    ids = [e.scan_id for e in after if e.kind == EventKind.OCR_RESULT]
    assert ids and ids[0] == 1


def test_empty_translation_keeps_the_block_hidden():
    p = pipeline_with(engine=EmptyEngine(), fps=60)
    p.start()
    p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.blocks)
    time.sleep(0.3)
    p.stop()
    assert p.stats()["mt_errors"] == 0
```

Adapt `pipeline_with` to build `BlockMemory("fast")` and pass `fps`; the fake OCR signature is `read(self, frame)` only. Keep `test_emitted_rects_are_screen_coordinates`, `test_pipeline_groups_stacked_lines_into_one_block` (real `BlockMemory`), `test_pipeline_emits_one_event_for_a_batch`, `test_cap_defers_blocks_instead_of_dropping_them`, engine-switch, and error-counter tests.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -q`
Expected: `TypeError` on the constructor (no `tracker` parameter) and missing `scan_rate`.

- [ ] **Step 3: Implement the scan loop**

Capture thread, per iteration: if `not enabled` sleep 0.05; if `self._ocr_busy.is_set()` sleep 0.016 and continue **without grabbing**; grab under the exclusion gate; if `frame is None` sleep the rest of the tick; emit `CAPTURE_HEALTH` when `health is not Health.OK` (no reset, no skip); compute `wake, gray = frame_changed(prev, frame)`; `interval = budget.interval(last_ocr_ms, empty_count, wake)`; if `now - last_ocr_end < interval` sleep the rest and continue; `self._interval = interval`, `self._scan_id += 1`, queue `(epoch, scan_id, frame, origin)`.

OCR thread: take the task, and **before touching `busy`** skip it when `epoch != self.epoch or not self.enabled`; otherwise `busy.set()`, then wrap the whole pass (`started = clock()`, `ocr.read`, `group_blocks`, `Box` conversion, `memory.observe(..., ttl_s=budget.ttl(self._interval))`) in `try/finally` where `finally` records `last_ocr_end`/`last_ocr_ms` and clears `busy` — so a failed or cancelled pass can never freeze the capture loop. On exception count and rate-limit the log. On success emit the complete snapshot with `scan_id`, `scan_interval_s=self._interval`, `scan_ms=self._last_ocr_ms`, set `empty_count = 0 if blocks else empty_count + 1`, and submit MT exactly as today. `set_regions`, `adopt_engine`, `set_enabled` and `stop` must call `self.memory.reset()` (a new epoch must not inherit blocks) and `self._wake_prev = None` (a new region has a new frame shape).

Delete `track.py` and `tests/test_track.py`; remove the `Decision`/crop path in `_run_ocr` **and the Task 3 complete-snapshot bridge** (snapshots now come from the OCR pass only). Delete `RapidOcrEngine._read_crops` and the `rects` argument of `ocr.read` (update `tests/test_ocr.py` and the fake OCRs). Update the other call sites to the new constructor (`TemporalTracker(...)` argument removed, including `src/ocrtrans/bench/__main__.py`) and `app.py` to `Pipeline(capture, RapidOcrEngine(), TranslationService(...), events.put, memory=BlockMemory(cfg.stability_profile), budget=ScanBudget(cfg.capture_fps), gate=AffinityGate(), fps=cfg.capture_fps)`, `pipeline.set_profile(...)`, `pipeline.set_fps(...)`.

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass, no skips beyond the `real`/`gpu` ones.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: text-first scan loop with one OCR pass in flight"
```

---

### Task 5: Bounded tiled rescue

**Files:**
- Modify: `src/ocrtrans/ocr.py` (add `plan_tiles`, `merge_boxes`)
- Modify: `src/ocrtrans/pipeline.py` (`_read_tiled`, `_tile_allowed`)
- Test: `tests/test_ocr.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `ocr.plan_tiles(width: int, height: int, max_side: int = 1280, overlap: int = 64, max_tiles: int = 2) -> list[Rect]` (region-local); `ocr.merge_boxes(boxes: list[Box], threshold: float = 0.5) -> list[Box]`; `Pipeline.tile_cooldown_s: float = 2.0` constructor keyword.

- [ ] **Step 1: Write the failing tests**

```python
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
```

Pipeline test: `pipeline_with(capture=FakeCapture(np.zeros((1440, 2560, 3), np.uint8)), ocr=WidthOcr(), region=Rect(0, 0, 2560, 1440), fps=60)` where `WidthOcr.read` returns `[]` for a frame at least 2560 px wide and one box for any narrower (tile) frame; assert an `OCR_RESULT` with that box arrives within 4 s (`test_tiled_rescue_finds_text_a_single_pass_missed`) — the first tiled pass cannot run before the 2 s cooldown.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr.py -q`
Expected: `ImportError` for `plan_tiles`.

- [ ] **Step 3: Implement**

`plan_tiles`: `cols = max(1, min(max_tiles, ceil(width / max_side)))`, same for rows; tile size `ceil(width/cols)` x `ceil(height/rows)`; extend each tile by `overlap` on its right/bottom (clamped) so seams are covered; return region-local rects **row-major (`for row: for col`)**, so the first tile is the top-left.

`Pipeline._read_tiled(frame)`: crop each tile, `ocr.read(crop)`, offset every box rect by the tile origin, then `merge_boxes`, then `group_blocks` as usual. In the OCR pass, if the single pass found no boxes, the region's long side is > 1600, `now - last_tiled_at >= tile_cooldown_s`, and the previous pass was not tiled, run the tiled read instead; increment `self._tiled`, set `last_tiled_at`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr.py tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/ocr.py src/ocrtrans/pipeline.py tests/test_ocr.py tests/test_pipeline.py
git commit -m "feat: bounded tiled rescue for large regions"
```

---

### Task 6: Stats, status line, README

**Files:**
- Modify: `src/ocrtrans/pipeline.py` (`stats`)
- Modify: `src/ocrtrans/app.py` (`status`)
- Modify: `src/ocrtrans/ui/control.py:134-147`
- Modify: `README.md`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: stats keys `scan_rate` (float, passes/second over the last 5 s), `tiled` (int), `last_scan_ms` (float), and `app.status` keys `rendered` (int, `OverlayWindow.shown`) and `hidden_stale` (int, removals counted by `OverlayState`).

- [ ] **Step 1: Write the failing test**

```python
def test_stats_report_scan_rate_and_last_pass_duration():
    p = pipeline_with(fps=60)
    p.start()
    assert p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT)
    time.sleep(0.3)
    stats = p.stats()
    p.stop()
    assert stats["scan_rate"] > 0
    assert stats["last_scan_ms"] > 0
    assert "tiled" in stats
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py::test_stats_report_scan_rate_and_last_pass_duration -q`
Expected: `KeyError: 'scan_rate'`.

- [ ] **Step 3: Implement**

`Pipeline`: keep a `collections.deque` of pass timestamps; `scan_rate = len(window) / 5.0` after trimming to the last 5 s; `last_scan_ms = self._last_ocr_ms`; `tiled = self._tiled`. `app.status` adds `"rendered": overlay.shown` and `"hidden_stale": overlay.state.hidden_stale`. `ui/control.set_status` second line becomes
`OCR {passes} pass ({rate:.1f}/s)   text {blocks}   shown {rendered}{error_text}   tiled {tiled}` and the third line keeps MT queue plus `p50/p95`, appending `scan {last_scan_ms:.0f} ms` and `hidden {hidden_stale}`.

`README.md`: replace the "how it schedules" description with the text-first loop (one pass in flight, budget table, TTL, exact text), update the metrics table header with `scan/s`, `hidden`, and `last_scan`, and note that motion no longer triggers extra OCR work.

- [ ] **Step 4: Run the suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/pipeline.py src/ocrtrans/app.py src/ocrtrans/ui/control.py README.md tests/test_pipeline.py
git commit -m "feat: report scan rate and stale-hide stats"
```

---

### Task 7: Video-background integration test

**Files:**
- Create: `tests/test_video_background.py`

**Interfaces:**
- Consumes: `Pipeline`, `OverlayState`, `RapidOcrEngine`, `BlockMemory`, `ScanBudget`, `TranslationService`.

- [ ] **Step 1: Write the test**

Synthetic frames: a 900x200 canvas filled with a random noise background regenerated on every grab (moving video), plus a fixed subtitle drawn with `C:/Windows/Fonts/segoeui.ttf` at 34 px. Drive `Pipeline` with `RapidOcrEngine()` on CPU and a fake engine returning `"VI " + text`. Feed every event into one `OverlayState` and record rendered items.

Assertions:
1. Within 6 s the subtitle "Open the gate, traveler." is rendered (`OverlayState` returns an item containing that source), and its `block_id` is recorded.
2. Then replace the subtitle with "The gate is locked." drawn at the **same left edge, baseline and font size**, and grab again; within 6 s an item for the new text is rendered.
3. From the first rendered item that contains the new text onward, no rendered item may pair the **old `block_id` with the old `source_text`** (the same rect keeps the same id by design, so the id alone proves nothing). (Before that first new item the old cover is allowed to remain: that is the deliberate scan-interval gap, and the new text is visible next to it.)

```python
def test_moving_background_subtitle_translates_and_never_shows_the_old_text():
    ...
    first_new = next(i for i, item in enumerate(items) if "locked" in item.source_text.lower())
    tail = items[first_new:]
    assert all("traveler" not in item.source_text.lower() for item in tail)
    assert all(
        not (item.block_id == old_block_id and item.source_text.lower() == old_source.lower())
        for item in tail
    )
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python -m pytest tests/test_video_background.py -q`
Expected: PASS. This is an acceptance test for the whole plan: a failure here means a real stale-translation or moving-background bug, so fix the pipeline, not the assertions (the only allowed adjustment is the font size if RapidOCR cannot read the synthetic subtitle at all).

- [ ] **Step 3: Commit**

```bash
git add tests/test_video_background.py
git commit -m "test: subtitle over a moving background never keeps the old translation"
```

---

## Self-review notes

- Spec §2 goals -> Tasks 1-7; anti-goals (stale paint, similarity, unbounded work, fallback silence) -> Tasks 3, 4, 7.
- Spec §4.2/4.3 -> Task 4; §4.4 -> Task 1; §4.5 -> Task 2; §4.6 -> Task 3; §4.7 -> Task 6; §4.3 tiling -> Task 5; §5 -> Tasks 3-4; §6 -> tests in every task; §8 -> Task 4 (removals) and Task 6 (README).
- Argos and the LibreTranslate note are a separate plan: `docs/superpowers/plans/2026-10-24-argos-engine.md` (independent subsystem).
