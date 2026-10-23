# Control Window, Paragraph Blocks and Real Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the app a visible control window, translate whole paragraphs instead of single OCR lines, keep painted text inside the source box, batch translation, and add opt-in tests against the real Gunfire Reborn window.

**Architecture:** The pipeline gains a paragraph-grouping step before the tracker; `line_px` travels from OCR boxes through the tracker and render items into `fit_text`. The MT thread drains a batch and emits one event. A new `ControlWindow` owns every setting, Start/Stop and status, and `app.py` wires it; the pipeline is started once and paused/resumed afterwards.

**Tech Stack:** Python 3.13, PyQt6, RapidOCR (ONNX/CUDA), transformers NLLB/EnViT5, `windows-capture` 2.0.1, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-control-window-design.md`

## Global Constraints

- Windows 11 only; Python 3.13; PyQt6; no new runtime dependencies.
- All coordinates are physical pixels; Qt maps through `devicePixelRatioF()`.
- Overlay must never exceed the source rectangle (no width widening, no vertical growth).
- Code, tests, docs and commit messages in English; chat in Vietnamese.
- No process injection, memory reads or global keyboard hooks (RegisterHotKey only).
- Config lives at `%LOCALAPPDATA%\OCRTrans\config.json`.
- Default `pytest` must stay fast: GPU and real tests are opt-in.

## Review Focus

1. Game in exclusive fullscreen: WGC window capture fails; real capture test must skip with a clear reason, not crash.
2. HDR frame with 4 channels: OCR and fit must receive BGR without a color/stride error.
3. Dialogue paragraph that grows by one line: tracker must keep the same block id (containment match) instead of flickering.
4. Long Vietnamese line at the font floor: ellipsis inside the source rect, never painted over the neighbor.
5. Engine switch while a batch is translating: switch defers, no half-translated batch is emitted.

---

### Task 1: Paragraph blocks reach the pipeline

**Files:**
- Modify: `src/ocrtrans/geometry.py` (`Box`), `src/ocrtrans/ocr.py` (`Block`), `src/ocrtrans/pipeline.py` (`_run_ocr`)
- Test: `tests/test_pipeline.py`, `tests/test_ocr.py`

**Interfaces:**
- Produces: `Box(rect: Rect, text: str, confidence: float, line_px: int = 0)`; `Block.line_px -> int`; pipeline emits one `BlockView` per paragraph.

- [ ] **Step 1: Write the failing test**

```python
def test_pipeline_groups_stacked_lines_into_one_block():
    ocr = FakeOcr()
    ocr.boxes = [
        Box(Rect(10, 10, 180, 20), "Open the gate", 0.9),
        Box(Rect(10, 32, 180, 20), "traveler.", 0.9),
    ]
    p = pipeline_with(ocr=ocr)
    p.start()
    event = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.blocks)
    assert len(event.blocks) == 1
    assert event.blocks[0].source_text == "Open the gate traveler."
    p.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py::test_pipeline_groups_stacked_lines_into_one_block -v`
Expected: FAIL (two blocks).

- [ ] **Step 3: Implement**

`geometry.Box` gains `line_px: int = 0`. `ocr.Block` gains:

```python
@property
def line_px(self) -> int:
    return _median([line.rect.h for line in self.lines]) if self.lines else self.rect.h
```

`pipeline._run_ocr` groups before observing:

```python
from .ocr import group_blocks
...
grouped = group_blocks(boxes)
boxes = [
    Box(rect=b.rect, text=b.text, confidence=min(l.confidence for l in b.lines), line_px=b.line_px)
    for b in grouped
]
blocks = self.tracker.observe(boxes, self.clock())
```

Crop path keeps the existing block `line_px` (crop boxes carry `line_px=0`).

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py tests/test_ocr.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/geometry.py src/ocrtrans/ocr.py src/ocrtrans/pipeline.py tests/test_pipeline.py
git commit -m "fix: group OCR lines into paragraph blocks before tracking"
```

---

### Task 2: Tracker carries line height and survives paragraph growth

**Files:**
- Modify: `src/ocrtrans/track.py`
- Test: `tests/test_track.py`

**Interfaces:**
- Consumes: `Box.line_px`.
- Produces: `TrackBlock.line_px: int`; matching accepts `iou >= 0.5` or containment `inter / min(area) >= 0.7`.

- [ ] **Step 1: Write the failing tests**

```python
def test_block_keeps_line_height():
    t = TemporalTracker()
    t.observe([Box(Rect(0, 0, 200, 20), "A", 0.9, line_px=18)], 0.0)
    assert t.blocks()[0].line_px == 18


def test_grown_paragraph_keeps_its_id():
    t = TemporalTracker()
    t.observe([Box(Rect(0, 0, 200, 20), "A", 0.9)], 0.0)
    first = t.blocks()[0].id
    t.observe([Box(Rect(0, 0, 200, 60), "A B C", 0.9)], 0.07)
    assert t.blocks()[0].id == first
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_track.py -q`
Expected: FAIL (`line_px` missing; grown block gets a new id).

- [ ] **Step 3: Implement**

Add `line_px: int = 0` to `TrackBlock`. In `observe`, after matching or creating: `if box.line_px: block.line_px = box.line_px`. Replace the `best_iou = 0.5` loop with a helper:

```python
def _match(a: Rect, b: Rect) -> bool:
    if iou(a, b) >= 0.5:
        return True
    inter_w = min(a.right, b.right) - max(a.x, b.x)
    inter_h = min(a.bottom, b.bottom) - max(a.y, b.y)
    if inter_w <= 0 or inter_h <= 0:
        return False
    smaller = min(a.w * a.h, b.w * b.h)
    return smaller > 0 and (inter_w * inter_h) / smaller >= 0.7
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_track.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/track.py tests/test_track.py
git commit -m "fix: keep paragraph ids and line height across tracking"
```

---

### Task 3: Text stays inside the source box

**Files:**
- Modify: `src/ocrtrans/fit.py`, `src/ocrtrans/overlay.py`, `src/ocrtrans/pipeline.py` (`BlockView`, `RenderItem`), `src/ocrtrans/config.py`
- Test: `tests/test_fit.py`

**Interfaces:**
- Produces: `fit_text(text, source, screen, measure, *, line_px=0, min_font_px=12, obstacles=()) -> FitResult`; `BlockView.line_px`, `RenderItem.line_px`.

- [ ] **Step 1: Write the failing tests**

```python
def test_font_follows_line_height_not_block_height():
    one_line = fit_text("Hello", Rect(0, 0, 300, 20), Rect(0, 0, 1000, 800), MonoMeasure(), line_px=20)
    paragraph = fit_text("Hello", Rect(0, 0, 300, 100), Rect(0, 0, 1000, 800), MonoMeasure(), line_px=20)
    assert one_line.font_px == paragraph.font_px


def test_fit_never_leaves_the_source_box():
    result = fit_text(
        "a very long translated line that must wrap many times",
        Rect(50, 50, 200, 24), Rect(0, 0, 1000, 800), MonoMeasure(), line_px=24,
    )
    assert result.rect.x == 50 and result.rect.w <= 200
    assert result.rect.h <= 24
    assert result.truncated


def test_shrinks_instead_of_growing():
    # replaces test_grows_before_shrinking_or_truncating from the old suite
    result = fit_text("một hai ba bốn năm sáu", Rect(0, 500, 200, 40), Rect(0, 0, 1920, 1080), MonoMeasure())
    assert result.font_px < 32 and result.rect.h <= 40
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_fit.py -q`
Expected: FAIL (paragraph font differs; rect grows).

- [ ] **Step 3: Implement**

`fit_text`: `base_font = max(min_font_px, min(round((line_px or source.h * 0.8) * 0.8), source.h))`; `width = source.w`; `max_h = source.h`; drop `max_growth` and the widening/growing loops; keep the shrink loop, floor and ellipsis. Remove `max_growth` from `Config` and from the overlay call. Add `line_px` to `BlockView`/`RenderItem`, fill it in `_ocr_loop`/`_submit_mt`/`_translate`, and pass `item.line_px` from `OverlayWindow._fit`.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_fit.py tests/test_lifecycle.py tests/test_e2e_smoke.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/fit.py src/ocrtrans/overlay.py src/ocrtrans/pipeline.py src/ocrtrans/config.py tests/test_fit.py
git commit -m "fix: size translated text from the source line height and keep it in place"
```

---

### Task 4: Batch translation and a per-pass cap

**Files:**
- Modify: `src/ocrtrans/translate.py`, `src/ocrtrans/pipeline.py`
- Test: `tests/test_translate.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `TranslationService.translate_many(texts: list[str]) -> list[tuple[str, bool]]`.
- Produces: `Pipeline._mt_loop` drains at most 8 tasks and emits one `TRANSLATION_RESULT`.

- [ ] **Step 1: Write the failing tests**

```python
class CountingEngine:
    name, kind = "counting", "local"

    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        self.calls.append(tuple(texts))
        return [t.upper() for t in texts]


def test_translate_many_uses_one_engine_call_and_caches():
    engine = CountingEngine()
    service = TranslationService(engine, TranslationCache())
    first = service.translate_many(["a", "b"])
    second = service.translate_many(["a", "b"])
    assert engine.calls == [[("a", "b")]]
    assert first == [("A", False), ("B", False)]
    assert second == [("A", True), ("B", True)]


def test_pipeline_emits_one_event_for_a_batch():
    p = pipeline_with(ocr=FakeOcr(boxes=[
        Box(Rect(10, 10, 100, 20), "one", 0.9),
        Box(Rect(10, 40, 100, 20), "two", 0.9),
        Box(Rect(10, 70, 100, 20), "three", 0.9),
    ]))
    p.start()
    event = p.wait_event(lambda e: e.kind == EventKind.TRANSLATION_RESULT and len(e.items) == 3)
    assert event is not None
    p.stop()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_translate.py tests/test_pipeline.py -q`
Expected: FAIL (`translate_many` missing; events carry one item).

- [ ] **Step 3: Implement**

`translate_many`: normalize, serve cache hits, batch the unique misses in one `engine.translate(list)` call, except engines with a `primary` attribute (`OnlineRouter`) which are called per item; on failure fall back to `self.fallback` for the whole batch, else return `""` for those items. Pipeline `_mt_loop` drains up to 8, calls `translate_many`, skips empty results after `self._last_mt.pop(block_id, None)`, emits one event with all items. `_ocr_loop` submits at most 12 stable blocks per pass, largest area first, text at least 3 characters, before `_last_mt` is marked.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_translate.py tests/test_pipeline.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/translate.py src/ocrtrans/pipeline.py tests/test_translate.py tests/test_pipeline.py
git commit -m "feat: translate OCR blocks in batches"
```

---

### Task 5: Control window

**Files:**
- Create: `src/ocrtrans/ui/control.py`, `tests/conftest.py`
- Modify: `src/ocrtrans/app.py`, delete `src/ocrtrans/ui/settings.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Produces: `ControlWindow(config: Config, on_start, on_stop, on_pick_region, on_apply, status_provider)` with `read_config() -> Config` and `set_status(dict) -> None`.
- Produces: `qapp` session fixture in `tests/conftest.py` returning one `QApplication` for the test session.
- Consumes: `pipeline.stats()`, `pipeline.set_enabled`, `pipeline.set_engine`, `pipeline.set_regions`.

- [ ] **Step 1: Write the failing tests**

```python
def test_control_window_reflects_config(qapp):
    cfg = Config(engine="envit5", capture_fps=20, min_font_px=16)
    window = ControlWindow(cfg, lambda: None, lambda: None, lambda: None, lambda c: None, dict)
    window.engine.setCurrentText("google")
    window.fps.setValue(25)
    assert window.read_config().engine == "google"
    assert window.read_config().capture_fps == 25
    window.close()


def test_start_button_calls_callback(qapp):
    calls = []
    window = ControlWindow(Config(), lambda: calls.append("start"), lambda: None,
                           lambda: None, lambda c: None, dict)
    window.start_button.click()
    assert calls == ["start"]
    window.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_control.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

`ControlWindow` with the fields from the spec, `read_config` mutating and returning the config, a Start/Stop button that toggles label and callback, region button showing `n vùng`, status label refreshed by a 500 ms `QTimer` from `status_provider`, and `closeEvent` calling `on_close`. In `app.py`: show the window at startup, start nothing; Start builds the engine with status `Đang tải model…`, starts the pipeline once, shows the overlay; Stop calls `set_enabled(False)` and clears the overlay; `on_apply` saves the config, sets fps/profile/regions and switches engine when running; `Ctrl+Alt+S` raises the window. Remove the settings dialog import and file.

- [ ] **Step 4: Run tests and smoke**

Run: `.venv/Scripts/python -m pytest tests/test_control.py tests/test_e2e_smoke.py -q`
Run: `.venv/Scripts/python -m ocrtrans.app --smoke` (add a `--smoke` path that presses Start and quits after 4 s)
Expected: PASS, exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/ui/control.py src/ocrtrans/app.py tests/test_control.py
git rm src/ocrtrans/ui/settings.py
git commit -m "feat: add a control window with all settings and start/stop"
```

---

### Task 6: Real game tests

**Files:**
- Create: `tests/real/test_game.py`, `tests/conftest.py`
- Modify: `src/ocrtrans/capture.py` (`WgcBackend` window target), `pyproject.toml` (markers)
- Test: the file itself

**Interfaces:**
- Produces: `WgcBackend(monitor_index=None, window_name=None, window_hwnd=None)`.
- Produces: `--run-real` option and `real` marker.

- [ ] **Step 1: Write the failing test (skip plumbing first)**

```python
# tests/real/test_game.py
def test_game_window_is_found(game_window):
    assert game_window.hwnd > 0
    assert "gunfire" in game_window.title.lower()
```

`tests/conftest.py` skips `real` tests without `--run-real` and the fixture skips when the window is missing or minimized.

- [ ] **Step 2: Run it both ways**

Run: `.venv/Scripts/python -m pytest -m real tests/real -q`
Expected: 1 skipped (needs --run-real).
Run: with the game open, `.venv/Scripts/python -m pytest -m real tests/real --run-real -q`
Expected: PASS on `test_game_window_is_found`.

- [ ] **Step 3: Add the remaining real tests and the window backend**

Tests: capture (`WindowsCapture(window_hwnd=...)`, assert 3/4 channels and luma), OCR (real `RapidOcrEngine`, skip when the screen has no text), translate (real engine, non-empty, digits preserved), pipeline 15 s (real `CaptureManager(WgcBackend(window_hwnd=...))`, all threads alive, `ocr_passes >= 1`). `WgcBackend` passes `window_hwnd`/`window_name` to `WindowsCapture`.

- [ ] **Step 4: Run the full real suite**

Run: `.venv/Scripts/python -m pytest -m real tests/real --run-real -q`
Expected: PASS with the game open in borderless mode; printed OCR/MT timings.

- [ ] **Step 5: Commit**

```bash
git add tests/real/test_game.py tests/conftest.py src/ocrtrans/capture.py pyproject.toml
git commit -m "test: add opt-in tests against the running game window"
```

---

### Task 7: Docs, benchmark, whole-branch review

**Files:**
- Modify: `README.md`, `bench/report.md` (regenerated)

- [ ] **Step 1: Update README** with the control window, the real-test command, the font/box rule and the known NLLB quality limit.
- [ ] **Step 2: Re-run benchmarks** `.venv/Scripts/python -m ocrtrans.bench --engine all --quality --perf --ocr --no-comet` and record the numbers.
- [ ] **Step 3: Run everything** `.venv/Scripts/python -m pytest -q`, `.venv/Scripts/python -m pytest -m gpu -q`, and the real suite with the game open.
- [ ] **Step 4: Whole-branch review** with a fresh reviewer; fix real defects.
- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document the control window, real tests and text-fit rules"
```
