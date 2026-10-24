# Truncated captions, primary-monitor capture, and a CPU/GPU installer

Date: 2026-10-24
Status: draft for approval

## Intent

Three independent changes, plus one follow-up that belongs to the third, each shipped
on its own so a failure in one does not block the others. In order: stop cutting
translated text, capture the monitor the app actually uses, and make the public repo
installable without a CUDA GPU (with an engine fallback so the CPU install starts
working instead of showing an error).

Evidence for the first: 27 `block N truncated at the font floor` warnings in the last
session's `app.log` (487 lines, 62 stats samples). Evidence for the second:
`app.py:115,120` pass `monitor_index=1` to a library whose index is one-based into
`EnumDisplayMonitors` order, while `window-capture` also exposes the primary monitor
by identity. Evidence for the third: `requirements.txt` pins
`torch==2.14.1+cu130` and `onnxruntime-gpu`, so anyone without an NVIDIA GPU cannot
install the project at all.

## Goals

1. A translated block is no longer ellipsized when a little more vertical room fixes
   it, without covering the next OCR block.
2. The WGC capture targets the primary monitor by identity, not by a guessed index.
3. A fresh Windows machine can install and run the app on CPU only (Argos engine), and
   a GPU machine still gets the CUDA path and NLLB, with one documented command each.
4. Every behaviour change above has a test that fails without it.

## Non-goals

- Multi-monitor support. Regions live on the primary screen (`app.py` uses
  `primaryScreen()` and `monitor_id` is always `"0"`); adding it would touch regions,
  the overlay and the picker. Out of scope.
- Any change to OCR or translation quality, model choice, or the pipeline.
- A PyInstaller/Squirrel installer, code signing, or auto-update. GitHub Release
  artifacts for torch exceed the 2 GB per-file limit.
- Reducing the CUDA download size.

## Part 1: grow the cover instead of cutting the text

Today `fit_text` (`fit.py:122`) caps the painted height at `source.h`, so a longer
Vietnamese line shrinks to `min_font_px` and the tail is replaced with `…`.

**Approach chosen: bounded downward growth.**

- `fit_text` gains `limit_bottom: int | None = None`. The allowed height becomes
  `max_h = max(1, (limit_bottom or source.bottom) - source.y)`. Everything else stays:
  same wrap, same font search, same ellipsis as the last resort.
- `_Layout.cover` grows to the fitted height: `Rect(item.rect.x, item.rect.y,
  item.rect.w, fitted_h)`, so the background hides whatever moves under it and no
  English shows through.
- The overlay computes, per block, `limit_bottom = min(window.bottom, item.rect.bottom
  + GROW_PX)` and, when another block starts inside that band, the nearer
  `next_block_top - 4`. Blocks are painted in list order, so overlap with another
  cover would flicker; the cap prevents it.
- `set_regions` reserves the room in the window: the bottom margin becomes
  `MARGIN_PX + GROW_PX`, where `GROW_PX = 2 * min_font_px + 8` (two floor lines). The
  window therefore never resizes because of a grown cover.
- A block at the very bottom of the screen still truncates, because there is nowhere
  to grow; the warning stays for that case.

**Rejected:** shrinking below `min_font_px` (breaks the setting the user just got);
tightening line height and padding only (buys at most one line, leaves the 27
warnings).

**Tests** (`tests/test_fit.py`, `tests/test_overlay_cover.py`):

- a text that truncates at `source.h` fits untruncated when `limit_bottom` allows two
  more lines, and its rect never passes `limit_bottom`;
- no `limit_bottom` keeps today's result (regression);
- overlay: a two-line translation yields a cover taller than the source block and
  inside the window; a block with another block 10 px below does not grow past it.

## Part 2: capture the primary monitor by identity

`windows_capture.WindowsCapture(monitor_index=N)` takes a **one-based** index into
`EnumDisplayMonitors` order. Index 1 is the first monitor returned, which is usually
but not always the primary one; on a laptop with an external display it can be the
wrong screen, and the crop (absolute screen coordinates) then reads garbage.

**Approach chosen: compute the primary index once, at backend construction.**

- New pure helper in `capture.py`:
  `primary_monitor_index(monitors: list[tuple[Rect, bool]], fallback: int = 1) -> int`
  returning the one-based position of the entry whose `primary` flag is true.
- A thin Win32 reader `enumerate_monitors() -> list[tuple[Rect, bool]]` using
  `EnumDisplayMonitors` + `GetMonitorInfoW` (`MONITORINFOF_PRIMARY`), not Qt, so the
  ordering matches the capture library's own enumeration.
- `app.py` passes `WgcBackend(monitor_index=primary_monitor_index(enumerate_monitors()))`
  in both places. If enumeration fails, it logs once and falls back to 1, which is
  today's behaviour.
- Guard, because the library's ordering cannot be observed on a one-monitor machine: the
  first delivered frame's width and height are compared with the chosen monitor's rect;
  on a mismatch the backend logs an error, rebuilds once with index 1, and records
  `monitor_mismatch` in the stats dict. Two identical monitors are still
  indistinguishable; the log line names the rect so a user can see it.

**Rejected:** per-region lazy monitor selection (the app has no multi-monitor model);
`Monitor.primary()` from the library (the installed Python package exposes no
`Monitor` class — checked with `dir(windows_capture)`).

**Tests** (`tests/test_capture.py`): the pure helper with primary first, primary last,
no primary flag, and an empty list (fallback). The frame-size guard is a small pure
function too (`monitor_size_matches(frame_wh, rect)`), tested with equal and unequal
sizes. The Win32 reader itself is covered by the existing `--smoke` run, which prints
the chosen index and rect.

## Part 3: install without a CUDA GPU

Checked fact: `argostranslate` depends on `stanza`, and `stanza` depends on `torch`, so
torch is installed in both variants. On Windows the default PyPI `torch` wheel is the
CPU build; the CUDA build only exists on `download.pytorch.org`. "CPU install" therefore
means: no CUDA index, no `onnxruntime-gpu`, no `transformers` (NLLB stays the GPU path).

**Approach chosen: two requirement files plus one PowerShell script.**

- `requirements-cpu.txt`: the same pinned packages with `onnxruntime` instead of
  `onnxruntime-gpu`, no `--extra-index-url` line, no `transformers`, and no explicit
  `torch` pin (it arrives as the CPU wheel through `stanza`). Argos works; NLLB does not
  because `transformers` is absent, which Part 4 handles.
- `install.ps1`:
  - `-Cpu` selects the CPU file; default is GPU.
  - creates `.venv` with `py -3.13 -m venv` (clear message when 3.13 is missing),
    upgrades pip, installs the chosen file, then `pip install -e . --no-deps` so
    `python -m ocrtrans.app` finds the `src/` package (today it only works because the
    existing venv already has an editable `.pth`).
  - prints the model-download note and `run.bat`.
  - `-DryRun` prints the commands instead of running them.
- `run.bat` stays as is. README gains one install command per variant and what the first
  start downloads (RapidOCR models ship with the wheel; Argos weights download on first
  use; NLLB weights only on the GPU path).
- No CI yet: the GPU install is ~3 GB and a runner cannot be verified from here.

**Rejected:** PyInstaller one-folder per variant (2 GB artifact limit, two 3 GB builds,
CUDA/onnxruntime DLL discovery is the fragile part); shipping model weights in the repo.

**Tests** (`tests/test_packaging.py`): both files parse; every non-comment line is
pinned (`==`); the CPU file has `onnxruntime`, no `onnxruntime-gpu`, no `+cu`, no extra
index URL and no `transformers`; the GPU file keeps the CUDA index URL. `install.ps1` is
syntax-checked with `[scriptblock]::Create` (skipped when PowerShell is absent), and
`-DryRun` is run once by hand.

## Part 4: fall back to Argos when NLLB cannot load

With `requirements-cpu.txt` there is no `transformers`, so the default engine (`nllb`)
cannot load and Start would only show an error.

**Approach chosen: fall back inside `build_local`.**

- `build_local("nllb")` imports `transformers` today at module import (`translate.py`), so
  the fallback lives where the engine is built: catch `ImportError`/`ModuleNotFoundError`
  for the NLLB path, log once (`nllb needs transformers; using argos`), and return the
  Argos engine instead.
- `app.load_engine` already turns exceptions into a status error; the fallback returns a
  working engine and sets `state["error"] = "nllb unavailable; using argos"` so the
  status line says why (single line, no dialog).
- GPU installs are unchanged: `transformers` is pinned there.

**Tests** (`tests/test_config.py`): with the NLLB import forced to raise,
`build_local("nllb")` returns an engine whose `name == "argos"`; with it importable the
NLLB engine is returned. `load_engine` is not tested directly (it loads models).

## Calls made (check these)

- Growth is capped at two floor lines and never overlaps another block. More room
  would need a smarter layout, not a bigger constant.
- The installer uses `py -3.13 -m venv`, not uv, to add no new dependency.
- The CPU variant keeps Argos (which needs torch through stanza) and drops
  `transformers`; NLLB stays the GPU path.
- The primary-monitor helper falls back to index 1, today's value, and the first frame
  is checked against the chosen monitor's size.
- `install.ps1` does the editable install (`pip install -e . --no-deps`) because the
  package lives under `src/`.

## Out of scope / known limits

- A block at the bottom edge of the screen still truncates.
- Multi-monitor capture, HiDPI, signed builds, auto-update.
- Two identical monitors cannot be told apart by the size guard; the index-order
  assumption stays a documented risk.
- The 4 `stanza: ... expects mwt` log lines stay (harmless, third-party logger).

## Tests summary

`pytest -q` must stay green. New: `tests/test_packaging.py`; additions to
`tests/test_fit.py`, `tests/test_overlay_cover.py`, `tests/test_capture.py`,
`tests/test_config.py` (engine fallback). The existing overlay geometry test
(`tests/test_overlay_cover.py:102`) needs its expected bottom margin updated for the
reserved growth room. The `--smoke` run must print the chosen monitor index and no
truncation warning for the sample caption.
