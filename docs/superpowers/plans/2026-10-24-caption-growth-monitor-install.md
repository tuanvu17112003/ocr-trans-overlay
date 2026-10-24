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
5. The CPU install is pinned, has no CUDA index, installs the package editable, and NLLB falls back to Argos instead of starting with an error — Tasks 3 and 4.

---

### Task 1: Grow the cover instead of cutting the text

**Files:**
- Modify: `src/ocrtrans/fit.py:104-143`
- Modify: `src/ocrtrans/overlay.py:55-60` (`MARGIN_PX` area), `_fit`, `_recompute`, `set_regions`
- Test: `tests/test_fit.py`, `tests/test_overlay_cover.py`

**Interfaces:**
- Produces: `fit_text(text, source, screen, measure, *, line_px=0, min_font_px=12, limit_bottom: int | None = None) -> FitResult`; the allowed height is `max(1, (limit_bottom or source.bottom) - source.y)`.
- Produces in `overlay.py`: `GROW_PX_LINES = 2`; `_Layout.cover` is `Rect(item.rect.x, item.rect.y, item.rect.w, max(item.rect.h, fitted_h))`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_fit.py` (uses the existing `MonoMeasure`):

```python
def test_limit_bottom_avoids_the_ellipsis():
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    box = Rect(0, 400, 200, 40)
    screen = Rect(0, 0, 1920, 1080)
    tight = fit_text(text, box, screen, MonoMeasure(), min_font_px=12)
    roomy = fit_text(text, box, screen, MonoMeasure(), min_font_px=12, limit_bottom=400 + 90)
    assert tight.truncated is True
    assert roomy.truncated is False
    assert roomy.rect.h > box.h


def test_limit_bottom_is_never_crossed():
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    roomy = fit_text(
        text, Rect(0, 400, 200, 40), Rect(0, 0, 1920, 1080), MonoMeasure(),
        min_font_px=12, limit_bottom=450,
    )
    assert roomy.rect.bottom <= 450


def test_no_limit_bottom_keeps_the_source_box():
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    box = Rect(0, 400, 200, 40)
    result = fit_text(text, box, Rect(0, 0, 1920, 1080), MonoMeasure(), min_font_px=12)
    assert result.rect.bottom <= box.bottom
```

In `tests/test_overlay_cover.py`:

```python
def test_a_grown_cover_covers_the_extra_lines(qapp):
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), queue.Queue())
    tall = make_item(text="một hai ba bốn năm sáu bảy tám chín mười", rect=Rect(100, 100, 200, 40))
    overlay.show_items([tall])
    layout = overlay._layouts[0]
    assert layout.cover.h >= 40
    assert layout.cover.bottom <= overlay.screen_rect.bottom
    overlay.close()


def test_a_block_below_caps_the_growth(qapp):
    cfg = Config()
    overlay = OverlayWindow(Rect(0, 0, 800, 600), cfg, queue.Queue())
    overlay.set_regions([Rect(80, 80, 240, 300)])
    above = make_item(text="một hai ba bốn năm sáu bảy tám chín mười", rect=Rect(100, 100, 200, 40))
    below = RenderItem(block_id=2, text="b", rect=Rect(100, 146, 200, 20), source_text="b", cached=False, line_px=20)
    overlay.show_items([above, below])
    assert overlay._layouts[0].cover.bottom <= 146
    overlay.close()


def test_damage_covers_the_grown_box(qapp):
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), queue.Queue())
    item = make_item(text="một hai ba bốn năm sáu bảy tám chín mười", rect=Rect(100, 100, 200, 40))
    damage = overlay._damage([], [item])
    assert damage.height() >= item.rect.h + overlay.GROW_PX
    overlay.close()
```

The existing geometry test at `tests/test_overlay_cover.py:102` expects the old bottom margin; update its expected rect to include the growth band.

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
- `WgcBackend` gains `self.monitor_mismatch: bool` and rebuilds once with index 1 when the first frame's size does not match.

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

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_capture.py -q`
Expected: FAIL — the names do not exist.

- [ ] **Step 3: Implement**

- `capture.py`: the two pure helpers; `enumerate_monitors()` using `ctypes` `EnumDisplayMonitors` + `GetMonitorInfoW` (return `[]` on any failure, so the caller falls back).
- `WgcBackend.__init__` keeps `monitor_index`; `_store_frame` on the first frame calls `monitor_size_matches((width, height), self._monitor_rect)`; on a mismatch and when not already retried, log an error, set `monitor_mismatch`, stop the session and rebuild with index 1. Keep the retry flag to one.
- `app.py`: `monitors = enumerate_monitors()`, `monitor_index = primary_monitor_index(monitors)`, log `capture monitor %s (%s)` once, and pass it to both `WgcBackend(...)` calls. Stats gain `monitor_mismatch` from the backend's attribute (default `False` when the backend lacks it).

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
- Produces: `requirements-cpu.txt` — same pins as `requirements.txt` but `onnxruntime`, no `transformers`, no `--extra-index-url`.
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
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_packaging.py -q`
Expected: FAIL — the files do not exist.

- [ ] **Step 3: Implement**

- `requirements-cpu.txt`: copy the pinned list, drop `--extra-index-url`, `transformers` and the `torch` line, replace `onnxruntime-gpu==1.31.0` with `onnxruntime==1.31.0`.
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
- Modify: `src/ocrtrans/app.py:52-58` (`build_local`) or `build_engine`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `build_engine(cfg)` returns `ArgosEngine` when `cfg.engine == "nllb"` and the NLLB build raises `ImportError`; logs once.

- [ ] **Step 1: Write the failing test**

```python
def test_nllb_falls_back_to_argos_without_transformers(monkeypatch):
    from ocrtrans import app as app_module

    def boom(name):
        raise ImportError("no transformers")

    monkeypatch.setattr(app_module, "build_local", boom)
    engine = app_module.build_engine(Config(engine="nllb"))
    assert engine.name == "argos"
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -q`
Expected: FAIL — the ImportError propagates.

- [ ] **Step 3: Implement**

In `build_engine`, wrap the `LOCAL_REPOS` branch:

```python
    if cfg.engine in LOCAL_REPOS:
        try:
            return build_local(cfg.engine)
        except ImportError:
            log.warning("%s needs transformers; using argos", cfg.engine)
            from .argos import ArgosEngine

            return ArgosEngine()
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
- Review Focus items map to tests: 1–3 in Task 1, 4 in Task 2, 5 in Tasks 3–4.
- The GPU install path is unchanged except `install.ps1`; no engine or pipeline behaviour moves.
