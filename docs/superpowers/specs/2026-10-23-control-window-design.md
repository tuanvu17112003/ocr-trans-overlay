# Control window, paragraph blocks and text-fit fixes

## Intent

The app runs headless: the only UI is an invisible overlay plus hotkey dialogs. The user
cannot tell whether it is running, cannot start or stop it, and cannot see the settings.
Reported symptoms: very slow, font sizes inconsistent, translation boxes larger than the
source and overlapping other text, translations wrong.

## Goals

- One always-visible control window with every user-facing setting, Start/Stop and live
  status.
- Font size derived from the source line height; the overlay never grows past the source
  box.
- Paragraph-level translation (already in the spec, never wired) instead of per line.
- Batch translation to cut end-to-end latency.

## Non-goals

- Tray icon (option B, rejected by the user).
- Translation preview window, installer, themes, multi-monitor UI.
- Raising NLLB-600M quality beyond correct paragraph grouping; no glossary in this pass.

## Approaches

- **A. One control window (chosen).** A single `QWidget` holding all settings, Start/Stop
  and status. Closing it quits. Smallest change that gives the user a visible app.
- **B. Window plus tray icon.** Same window, hidden to tray on close. Rejected: extra icon
  asset, extra lifetime state, nothing the user asked for.

## Design

### 1. Control window (`src/ocrtrans/ui/control.py`)

`ControlWindow(QWidget)`:

- Engine combo: `nllb`, `envit5`, `google`, `mymemory`, `deepl`, `gemini`.
- Key fields for Gemini and DeepL, plus the online-consent checkbox; enabled only when an
  online engine is selected.
- Stability profile (`safe`/`fast`), capture fps (5-30), region mode
  (`single`/`multi`/`monitor`), region picker button showing the current region count.
- Minimum font px (8-32), overlay opacity (0.6-1.0).
- Start/Stop button and a status line: engine, capture health, OCR passes, p50/p95, MT
  queue, degraded flag.
- Settings are written to `config.json` on Start and on close.
- Closing the window stops the pipeline and quits.

`app.py` wiring:

- The window is created before the pipeline starts and shown immediately.
- Start builds the engine (blocking; status says the model is loading), starts the
  pipeline and shows the overlay.
- Stop stops the pipeline and clears the overlay.
- Changing engine or regions while running applies through `pipeline.set_engine` and
  `pipeline.set_regions`.
- `Ctrl+Alt+S` raises the control window; the other hotkeys keep working.
- `ui/settings.py` and its usage are removed (superseded).

### 2. Paragraph blocks

- `pipeline._run_ocr` calls `group_blocks(boxes)` and feeds one `Box` per paragraph to the
  tracker: union rect, joined text, minimum confidence, `line_px` = median line height.
- `Box` gains `line_px: int = 0`; `Block` gains `line_px` (median of its line heights).
- Crop re-reads keep the existing block `line_px` (no det, one box per crop).

### 3. Font fit

- `fit_text(..., line_px=0)`: `base_font = clamp(round(line_px * 0.8), min_font_px,
  source.h)`; `line_px = 0` falls back to the current height-based estimate.
- No width widening and no vertical growth: `widths = [source.w]`, `max_h = source.h`.
- Shrink to `min_font_px`, then ellipsize and log (unchanged).
- `TrackBlock.line_px` and `BlockView.line_px` carry it; `RenderItem.line_px` reaches the
  overlay.

### 4. Throughput

- `TranslationService.translate_many(texts)` returns `list[tuple[str, bool]]`: cache per
  item, one batched engine call for the misses on local engines, per-item calls for
  online engines (their interface is single-text).
- `Pipeline._mt_loop` drains up to 8 queued blocks, translates them as one batch and
  emits a single `TRANSLATION_RESULT` with all items.
- At most 12 blocks are submitted per OCR pass, largest area first; text shorter than 3
  characters is skipped.

## Testing

- Unit: fit keeps the font proportional to `line_px` and never exceeds the source rect;
  `translate_many` batches and caches; the pipeline emits one event with N items; paragraph
  grouping reaches `BlockView`.
- Existing e2e smoke test still passes.
- Manual: open the window, Start/Stop, switch engine, pick a region, confirm painted boxes
  stay inside the source lines.

## Real integration tests

Unit tests keep their fakes: they are fast and cover logic. A second layer proves the
hardware path on the actual game.

- `tests/conftest.py`: `--run-real` option plus a `real` marker. Without the flag every
  real test is skipped with a reason (pytest `addoption` + `collection_modifyitems`).
- `tests/real/test_game.py`:
  - `game_window` fixture: `EnumWindows`, title contains `gunfire`, pick the top-level
    game window (not the launcher); skip when missing or minimized.
  - `test_capture_game_window`: real `WindowsCapture(window_hwnd=...)` first frame; assert
    it is not None, has 3 or 4 channels and luma above black.
  - `test_ocr_game_frame`: real `RapidOcrEngine`; assert boxes come back and record the
    time; skip with a reason when the current screen has no text.
  - `test_translate_game_text`: real engine from config; translate the first blocks;
    assert every result is non-empty, numbers are preserved, record the total time.
  - `test_pipeline_against_game`: real capture + OCR + MT + tracker for 15 s; assert all
    threads alive, `ocr_passes >= 1`, events flowing; fail if a thread died.
- Run with the game open in borderless mode:
  `pytest -m real --run-real -q`.

Measured on this machine (Gunfire Reborn settings screen, 2560x1440): WGC window capture
OK, mean 85.1; OCR 38 boxes / 37 blocks; first call 2.9 s including model load.

## Calls made from the builder check

1. **Stop is a pause.** `set_enabled(False)` plus overlay clear; threads and the model
   stay. Quit tears down. Start after the first one just re-enables.
2. **Nothing runs before Start.** The app opens the control window only; engine, capture
   and overlay start on the first Start, so no 2.4 GB download without consent.
3. **Batching** uses one `translate(list)` call for engines that return a list;
   `OnlineRouter` is single-text and is called per item.
4. **The 12-block cap** applies after dedupe and before `_last_mt` is marked, so overflow
   is deferred to the next pass instead of being dropped.
5. **Region picker replaces all regions.** Multi mode keeps using the stored list.
6. **A failed batch** falls back to the local engine once when configured; otherwise the
   failed items yield `""`, their `_last_mt` entry is cleared and they retry next pass.
7. **Paragraph matching** keeps `iou >= 0.5` and also accepts containment
   `intersection / min(area) >= 0.7`, so a paragraph that grows one line keeps its id.

## Risks

- NLLB-600M still mistranslates short game lines; grouping reduces but does not remove it.
- A batch of 8 uses more VRAM transiently than a single call.
