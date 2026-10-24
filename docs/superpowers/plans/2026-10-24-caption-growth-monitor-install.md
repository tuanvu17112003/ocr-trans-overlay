# Caption Growth, Primary Monitor and CPU Install — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop ellipsizing translated captions when a little vertical room fixes it, capture the primary monitor by identity instead of a guessed index, and let a machine without CUDA install and run the app on Argos.

**Architecture:** Four independent changes. `fit_text` gains a bottom limit and the overlay grows each cover inside a reserved band; capture asks Win32 which monitor is primary and verifies the first frame against it; a second requirements file plus `install.ps1` covers CPU installs; `build_engine` falls back from NLLB to Argos when `transformers` cannot be imported.

**Tech Stack:** Python 3.13, PyQt6, pytest (offscreen `qapp` fixture), PowerShell 5.1 for the installer.

**Spec:** `docs/superpowers/specs/2026-10-24-robustness-and-install-design.md`

## Global Constraints

- Windows only; no new dependencies; no new config fields.
- Coordinates are logical pixels; the target machine is `dpr = 1.0` (HiDPI out of scope).
- The overlay cover stays opaque: a grown cover must still hide the English under it.
- Every behaviour change lands with a test that fails without it.
- Tests run offline: `.venv/Scripts/python -m pytest -q`.

## Review Focus

1. A grown cover must never overlap another OCR block (paint order would flicker) — Task 1.
2. A block at the bottom edge of the screen still truncates, without a negative height — Task 1.
3. The overlay window reserves the growth band and the repaint damage includes it — Task 1.
4. The primary monitor is chosen by the `MONITORINFOF_PRIMARY` flag, the size guard falls back once, and the DXGI path is untouched — Task 2.
5. The CPU install is pinned, has no CUDA index, installs the package editable, and NLLB falls back to a **loaded** Argos instead of an engine that raises on every call — Tasks 3 and 4.

---

### Task 1: Grow the cover instead of cutting the text

**Files:**
- Modify: `src/ocrtrans/fit.py:104-143`
- Modify: `src/ocrtrans/overlay.py:55-60` (`MARGIN_PX` area), `_fit`, `_recompute`, `set_regions`
- Test: `tests/test_fit.py`, `tests/test_overlay_cover.py`

**Interfaces:**
- Produces: `fit_text(text, source, screen, measure, *, line_px=0, min_font_px=12, limit_bottom: int | None = None) -> FitResult`; the allowed height is `max(1, (limit_bottom or source.bottom) - source.y)`.
- Produces in `overlay.py`: `self.GROW_PX = 2 * config.min_font_px + 8` (set in `__init__`); `_Layout.cover` is `Rect(item.rect.x, item.rect.y, item.rect.w, max(item.rect.h, fitted_h))`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_fit.py` (uses the existing `MonoMeasure`; a 40 px box already fits two lines, so the tight fixture is 30 px):

```python
def test_limit_bottom_avoids_the_ellipsis():
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    box = Rect(0, 400, 200, 30)
    screen = Rect(0, 0, 1920, 1080)
    tight = fit_text(text, box, screen, MonoMeasure(), min_font_px=12)
    roomy = fit_text(text, box, screen, MonoMeasure(), min_font_px=12, limit_bottom=400 + 90)
    assert tight.truncated is True
    assert roomy.truncated is False
    assert roomy.rect.h > box.h


def test_limit_bottom_is_never_crossed():
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    roomy = fit_text(
        text, Rect(0, 400, 200, 30), Rect(0, 0, 1920, 1080), MonoMeasure(),
        min_font_px=12, limit_bottom=450,
    )
    assert roomy.rect.bottom <= 450


def test_no_limit_bottom_keeps_the_source_box():
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    box = Rect(0, 400, 200, 30)
    result = fit_text(text, box, Rect(0, 0, 1920, 1080), MonoMeasure(), min_font_px=12)
    assert result.rect.bottom <= box.bottom
```

In `tests/test_overlay_cover.py` (a 24 px block with that text grows under real Qt metrics):

```python
def test_a_grown_cover_covers_the_extra_lines(qapp):
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), queue.Queue())
    tall = make_item(text="một hai ba bốn năm sáu bảy tám chín mười", rect=Rect(100, 100, 200, 24))
    overlay.show_items([tall])
    layout = overlay._layouts[0]
    assert layout.cover.h > 24
    assert layout.cover.bottom <= overlay.screen_rect.bottom
    overlay.close()


def test_a_block_below_caps_the_growth(qapp):
    cfg = Config()
    overlay = OverlayWindow(Rect(0, 0, 800, 600), cfg, queue.Queue())
    overlay.set_regions([Rect(80, 80, 240, 300)])
    above = make_item(text="một hai ba bốn năm sáu bảy tám chín mười", rect=Rect(100, 100, 200, 24))
    below = RenderItem(block_id=2, text="b", rect=Rect(100, 146, 200, 20), source_text="b", cached=False, line_px=20)
    overlay.show_items([above, below])
    assert overlay._layouts[0].cover.h > 24
    assert overlay._layouts[0].cover.bottom <= 142
    overlay.close()


def test_damage_covers_the_grown_box(qapp):
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), queue.Queue())
    item = make_item(text="một hai ba bốn năm sáu bảy tám chín mười", rect=Rect(100, 100, 200, 24))
    damage = overlay._damage([], [item])
    assert damage.height() >= item.rect.h + overlay.GROW_PX
    overlay.close()
```

Two existing assertions move with the change: the geometry expectation at
`tests/test_overlay_cover.py:102` gains the bottom growth band, and the
`damage.height() <= 64` check at `:110` becomes
`<= item.rect.h + overlay.GROW_PX + 4`, keeping its "not the whole screen" intent.

- [ ] **Step 2: Run them and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_fit.py tests/test_overlay_cover.py -q`
Expected: FAIL — `fit_text` rejects `limit_bottom`; the cover never grows.

- [ ] **Step 3: Implement**

- `fit.py`: add the keyword, compute `max_h` from `limit_bottom`, keep everything else (`rect_for` already clamps with `max_h`).
- `overlay.py`:
  - `GROW_PX = 2 * min_font_px + 8` computed where the measure is available, and `set_regions` uses `bottom = min(full.bottom, box.bottom + MARGIN_PX + GROW_PX)`.
  - `_recompute` sorts nothing but computes per-item `limit_bottom`: `min(self.screen_rect.bottom, item.rect.bottom + GROW_PX)`, then the nearest `other.rect.y - 4` for any other item starting inside that band.
  - `_fit(item, limit_bottom)` passes it to `fit_text` and returns `cover=Rect(item.rect.x, item.rect.y, item.rect.w, max(item.rect.h, fit.rect.h))`.
  - `_damage` must include the growth band, because a grown cover paints below the source rect: grow each candidate rect by `GROW_PX` at the bottom (clamped to the window) before the union.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_fit.py tests/test_overlay_cover.py -q`
Expected: PASS.

- [ ] **Step 5: Whole suite, then commit**

Run: `.venv/Scripts/python -m pytest -q`

```bash
git add src/ocrtrans/fit.py src/ocrtrans/overlay.py tests/test_fit.py tests/test_overlay_cover.py
git commit -m "fix: let a translated caption grow instead of being cut"
```

---

### Task 2: Capture the primary monitor by identity

**Files:**
- Modify: `src/ocrtrans/capture.py`
- Modify: `src/ocrtrans/app.py:115,120`
- Test: `tests/test_capture.py`

**Interfaces:**
- Produces: `enumerate_monitors() -> list[tuple[Rect, bool]]` (Win32, rect in **physical** pixels, `True` for the primary). Physical because the size guard compares it with the captured frame, which is never scaled.
- Produces: `primary_monitor_index(monitors: list[tuple[Rect, bool]], fallback: int = 1) -> int`.
- Produces: `monitor_size_matches(frame_wh: tuple[int, int], monitor: Rect) -> bool`.
- Produces: `WgcBackend(..., monitor_rect: Rect | None = None)`; it gains `self.monitor_mismatch: bool`, a `self._retry_requested: bool` flag set from the frame callback, and `grab()` performs the single rebuild with index 1. Teardown never runs on the capture thread (see `capture.py:95-99`).

- [ ] **Step 1: Write the failing tests**

```python
from ocrtrans.capture import monitor_size_matches, primary_monitor_index
from ocrtrans.geometry import Rect


def test_primary_index_follows_the_flag_not_the_position():
    monitors = [(Rect(0, 0, 1920, 1080), False), (Rect(1920, 0, 2560, 1440), True)]
    assert primary_monitor_index(monitors) == 2


def test_primary_index_is_one_when_the_primary_is_first():
    assert primary_monitor_index([(Rect(0, 0, 1920, 1080), True)]) == 1


def test_primary_index_falls_back_without_a_flag():
    assert primary_monitor_index([(Rect(0, 0, 1920, 1080), False)], fallback=1) == 1
    assert primary_monitor_index([], fallback=1) == 1


def test_frame_size_must_match_the_monitor():
    assert monitor_size_matches((1920, 1080), Rect(0, 0, 1920, 1080)) is True
    assert monitor_size_matches((2560, 1440), Rect(0, 0, 1920, 1080)) is False
```

Note the rects here are the physical Win32 rects, so a 125% display does not trip the
guard.

```python
from ocrtrans.capture import needs_monitor_retry


def test_retry_happens_once_on_a_size_mismatch():
    rect = Rect(0, 0, 1920, 1080)
    assert needs_monitor_retry((2560, 1440), rect, already_retried=False) is True
    assert needs_monitor_retry((2560, 1440), rect, already_retried=True) is False
    assert needs_monitor_retry((1920, 1080), rect, already_retried=False) is False
    assert needs_monitor_retry((2560, 1440), None, already_retried=False) is False
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_capture.py -q`
Expected: FAIL — the names do not exist.

- [ ] **Step 3: Implement**

- `capture.py`: the two pure helpers plus `needs_monitor_retry(frame_wh, monitor_rect, already_retried) -> bool`; `enumerate_monitors()` using `ctypes` `EnumDisplayMonitors` + `GetMonitorInfoW` (return `[]` on any failure, so the caller falls back).
- `WgcBackend.__init__` takes `monitor_rect`; `_store_frame` only sets `self._retry_requested = needs_monitor_retry(...)` when the first frame mismatches, and logs once. `grab()` sees the flag before waiting on the next frame, tears down and rebuilds the session with index 1, sets `monitor_mismatch = True`, and clears the flag. Nothing stops or rebuilds the session from the capture thread.
- The two tests that build the backend with `WgcBackend.__new__` (`tests/test_capture.py:91-103`, `157-171`) get `backend._monitor_rect = None` and `backend._retry_requested = False` so the new fields exist.
- `app.py`: `monitors = enumerate_monitors()`, `monitor_index = primary_monitor_index(monitors)`, `monitor_rect` = that entry's rect, log `capture monitor %s (%s)` once, pass both to the two `WgcBackend(...)` calls.
- `status()` (`app.py:157-164`) gains `"monitor_mismatch": getattr(capture.main, "monitor_mismatch", False)`.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_capture.py -q`
Expected: PASS.

- [ ] **Step 5: Smoke, whole suite, commit**

Run: `.venv/Scripts/python -c "import sys; from ocrtrans.app import main; sys.argv=['ocrtrans','--smoke']; raise SystemExit(main())"`
Expected: the log prints `capture monitor 1` and `WgcBackend`, no traceback.

Run: `.venv/Scripts/python -m pytest -q`

```bash
git add src/ocrtrans/capture.py src/ocrtrans/app.py tests/test_capture.py
git commit -m "fix: capture the primary monitor by identity"
```

---

### Task 3: CPU requirements and the installer

**Files:**
- Create: `requirements-cpu.txt`
- Create: `install.ps1`
- Modify: `README.md` (Install section)
- Test: `tests/test_packaging.py`

**Interfaces:**
- Produces: `requirements-cpu.txt` with exactly: `numpy==2.5.4`, `opencv-python==5.0.0.93`, `windows-capture==2.0.1`, `rapidocr==3.10.0`, `onnxruntime==1.31.0`, `sentencepiece==0.2.2`, `argostranslate==1.11.0`, `requests==2.34.2`, `PyQt6==6.11.0`, `pytest==9.1.1`. No extra index, no `torch` line (it arrives as the CPU wheel through `stanza`), no `transformers`, no `onnxruntime-gpu`.
- Produces: `install.ps1 [-Cpu] [-DryRun]`.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _lines(name):
    raw = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in raw if line.strip() and not line.startswith("#")]


def test_gpu_requirements_keep_the_cuda_index():
    lines = _lines("requirements.txt")
    assert lines[0].startswith("--extra-index-url https://download.pytorch.org/whl/cu")
    assert any(line.startswith("torch==") and "+cu" in line for line in lines)
    assert any(line.startswith("onnxruntime-gpu==") for line in lines)


def test_cpu_requirements_have_no_cuda_and_no_transformers():
    lines = _lines("requirements-cpu.txt")
    assert not any(line.startswith("--extra-index-url") for line in lines)
    assert any(line.startswith("onnxruntime==") for line in lines)
    assert not any("onnxruntime-gpu" in line for line in lines)
    assert not any(line.startswith("transformers") for line in lines)
    assert all("+cu" not in line for line in lines)


def test_every_requirement_is_pinned():
    for name in ("requirements.txt", "requirements-cpu.txt"):
        for line in _lines(name):
            if line.startswith("--"):
                continue
            assert "==" in line, f"{name}: {line} is not pinned"


def test_installer_parses():
    script = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert "-Cpu" in script and "-DryRun" in script
    assert "pip install -e . --no-deps" in script


def test_installer_is_valid_powershell():
    if shutil.which("powershell") is None:
        pytest.skip("no PowerShell on this machine")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "$null = [scriptblock]::Create((Get-Content -Raw -LiteralPath 'install.ps1'))"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
```

(`import shutil, subprocess, pytest` at the top of the file.)

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_packaging.py -q`
Expected: FAIL — the files do not exist.

- [ ] **Step 3: Implement**

- `requirements-cpu.txt`: exactly the list in the Interfaces block (write it out; do not copy the GPU file).
- `install.ps1`:
  - param block with `[switch]$Cpu` and `[switch]$DryRun`;
  - check `py -3.13 --version`, else write an error naming the winget/store requirement and exit 1;
  - build `$steps` (create venv, `python -m pip install --upgrade pip`, `pip install -r requirements[-cpu].txt`, `pip install -e . --no-deps`, print the model note and `run.bat`);
  - `-DryRun` prints each step, otherwise runs it and stops on a non-zero exit.
- `README.md` Install: replace the manual venv block with the two commands (`.\install.ps1` for GPU, `.\install.ps1 -Cpu` for CPU) and say what the first start downloads.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_packaging.py -q`
Expected: PASS.

- [ ] **Step 5: Dry-run the installer by hand, commit**

Run: `powershell -NoProfile -File install.ps1 -Cpu -DryRun`
Expected: five printed steps, no changes on disk.

```bash
git add requirements-cpu.txt install.ps1 README.md tests/test_packaging.py
git commit -m "feat: CPU requirements and a one-command installer"
```

---

### Task 4: Fall back to Argos when NLLB cannot load

**Files:**
- Modify: `src/ocrtrans/app.py:61-95` (one guarded `build_local` helper), `load_engine` (status line)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `local_engine(name: str)` — the single place that calls `build_local`, with the NLLB→Argos fallback. `build_engine`, the online fallback (`app.py:81`) and the `fallback_to_local` path (`app.py:92`) all use it.
- Produces: the fallback engine is **loaded** before it is returned (`ArgosEngine.translate` raises until `load()`), and `load_engine` sets `state["error"] = "nllb unavailable; using argos"` when the returned engine's name differs from `cfg.engine`.

- [ ] **Step 1: Write the failing test**

```python
def test_nllb_falls_back_to_argos_without_transformers(monkeypatch):
    from ocrtrans import app as app_module
    from ocrtrans.argos import ArgosEngine

    loaded: list[bool] = []

    def boom(name):
        raise ImportError("no transformers")

    monkeypatch.setattr(app_module, "build_local", boom)
    monkeypatch.setattr(ArgosEngine, "load", lambda self: loaded.append(True))
    engine = app_module.build_engine(Config(engine="nllb"))
    assert engine.name == "argos"
    assert loaded == [True]
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -q`
Expected: FAIL — the ImportError propagates.

- [ ] **Step 3: Implement**

In `app.py`, above `build_engine`:

```python
def local_engine(name: str):
    try:
        return build_local(name)
    except ImportError:
        log.warning("%s needs transformers; using argos", name)
        from .argos import ArgosEngine

        engine = ArgosEngine()
        engine.load()
        return engine
```

Replace all three `build_local(...)` call sites with `local_engine(...)`, and in
`load_engine`, after a successful build:

```python
    if engine.name != config.engine:
        state["error"] = f"{config.engine} unavailable; using {engine.name}"
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Whole suite and commit**

Run: `.venv/Scripts/python -m pytest -q`

```bash
git add src/ocrtrans/app.py tests/test_config.py
git commit -m "feat: fall back to argos when nllb cannot load"
```

---

## Self-review notes

- Spec coverage: Part 1 → Task 1, Part 2 → Task 2, Part 3 → Task 3, Part 4 → Task 4.
- Review Focus items map to tests: 1–3 in Task 1, 4 in Task 2 (including `needs_monitor_retry`), 5 in Tasks 3–4.
- The GPU install path is unchanged except `install.ps1`; no engine or pipeline behaviour moves.
- External review round (production-plan-audit, NO-GO) closed: the tight fit fixture now actually truncates (h=30), the Argos fallback is loaded, every `build_local` call site goes through `local_engine`, the monitor rect is passed in, the retry is decided in `grab()` (never on the capture thread), and the two existing overlay assertions that the growth affects are named.
- Deliberately unverified: the library's monitor ordering on a two-monitor machine, `onnxruntime==1.31.0` on PyPI, and a real CPU-only install; each fails loudly rather than silently.
