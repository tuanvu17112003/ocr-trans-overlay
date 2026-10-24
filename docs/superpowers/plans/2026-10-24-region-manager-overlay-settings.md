# Region Manager and Overlay Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single "Pick region" button with a visual region manager (list, on-screen outlines, add/adjust/delete), drop the whole-monitor mode, and finish the control window as a dark production tool with overlay appearance settings.

**Architecture:** `Config.regions` stays the single source of truth (normalized per monitor). A new read-only outline window draws those regions; a temporary handle editor reuses the existing picker to edit one rect; the control window renders the list and calls back into `app.py`, which saves, applies and refreshes. No pipeline or overlay-rendering behaviour changes.

**Tech Stack:** Python 3.13, PyQt6, pytest with the existing `qapp` offscreen fixture.

**Spec:** `docs/superpowers/specs/2026-10-24-region-manager-overlay-settings-design.md`

## Global Constraints

- Windows only; PyQt6; no new dependencies.
- Coordinates are Qt logical pixels; the target machine is `dpr = 1.0`. HiDPI is out of scope.
- The overlay cover stays opaque; there is no opacity setting.
- Every region list change goes through `save_config` + `pipeline.set_regions` + `overlay.set_regions`.
- Colors and fonts come from the spec's tokens: bg `#12141A`, panel `#171A22`, border `#242833`, text `#E7EAF0`, muted `#8B93A5`, accent `#5EE0B0`, danger `#FF8F8F`.
- Tests run offline (`pytest -q`); Qt tests use the `qapp` fixture from `tests/conftest.py`.

## Review Focus

1. An old config with `region_mode: "monitor"` must load and run, not raise — Task 1.
2. Dragging a handle past the monitor edge or to zero size must clamp, never produce a degenerate rect — Task 4.
3. Deleting the last region must be impossible; Start with no region must be blocked with a hint — Task 2.
4. The outline must never be visible while the pipeline runs (a full-screen layered window over video) — Task 3 and Task 6.
5. Changing the overlay font while running must relayout with the new font, not keep the old metrics — Task 5.

---

### Task 1: Region modes, migration, clamping

**Files:**
- Modify: `src/ocrtrans/config.py:12`, `src/ocrtrans/config.py:53-61`
- Modify: `src/ocrtrans/regions.py:46-56`
- Modify: `src/ocrtrans/app.py:49`
- Test: `tests/test_config.py`, `tests/test_regions.py`

**Interfaces:**
- Produces: `REGION_MODES = frozenset({"single", "multi"})`; `clamp_rect(rect: Rect, monitor: Rect, min_size: int = 16) -> Rect` in `regions.py`; `RegionSet.from_config(cfg: Config, monitor: Rect) -> list[Rect]` (same signature, now clamps).

- [ ] **Step 1: Write the failing tests**

In `tests/test_regions.py`, replace `test_region_mode_monitor_is_full_screen` with:

```python
def test_single_mode_resolves_only_the_first_region():
    cfg = cfg_with("single", [{"monitor_id": "0", "x": 0.1, "y": 0.1, "w": 0.5, "h": 0.1},
                              {"monitor_id": "0", "x": 0.1, "y": 0.5, "w": 0.2, "h": 0.1}])
    assert RegionSet.from_config(cfg, Rect(0, 0, 1000, 1000)) == [Rect(100, 100, 500, 100)]


def test_multi_mode_resolves_every_region():
    cfg = cfg_with("multi", [{"monitor_id": "0", "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.1},
                             {"monitor_id": "0", "x": 0.0, "y": 0.5, "w": 0.2, "h": 0.1}])
    assert len(RegionSet.from_config(cfg, Rect(0, 0, 1000, 1000))) == 2


def test_a_region_outside_the_monitor_is_clamped():
    assert clamp_rect(Rect(900, 900, 400, 400), Rect(0, 0, 1000, 1000)) == Rect(600, 600, 400, 400)


def test_a_tiny_region_gets_the_minimum_size():
    assert clamp_rect(Rect(10, 10, 2, 2), Rect(0, 0, 1000, 1000)) == Rect(10, 10, 16, 16)
```

In `tests/test_config.py`:

```python
def test_old_monitor_config_loads_as_single(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"region_mode": "monitor", "regions": [{"monitor_id": "0", "x": 0.0, "y": 0.1, "w": 0.5, "h": 0.1}]}', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.region_mode == "single"
    assert len(cfg.regions) == 1


def test_monitor_mode_is_no_longer_writable(tmp_path):
    cfg = Config(region_mode="monitor")
    with pytest.raises(ValueError):
        save_config(cfg, tmp_path / "config.json")
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_config.py tests/test_regions.py -q`
Expected: FAIL — `clamp_rect` missing, `monitor` still accepted.

- [ ] **Step 3: Implement**

- `config.py`: `REGION_MODES = frozenset({"single", "multi"})`; in `load_config`, before constructing `Config`, map `"monitor"` to `"single"` (`raw["region_mode"]`).
- `regions.py`: add `clamp_rect` (offset into the monitor, then clamp x/y so the rect fits, then enforce `min_size`); delete the `monitor` branch in `RegionSet.from_config`; clamp each resolved rect with the monitor.
- `app.py`: `MODES = ["single", "multi"]`; the banner log keeps printing the applied rects.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_config.py tests/test_regions.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole suite and commit**

Run: `.venv/Scripts/python -m pytest -q`
Expected: PASS (any old `monitor` test already replaced).

```bash
git add src/ocrtrans/config.py src/ocrtrans/regions.py src/ocrtrans/app.py tests/test_config.py tests/test_regions.py
git commit -m "feat: drop the monitor region mode and clamp resolved regions"
```

---

### Task 2: Region list in the control window

**Files:**
- Modify: `src/ocrtrans/ui/control.py:25-122`
- Modify: `src/ocrtrans/app.py:299` (constructor call only; behaviour wiring is Task 6)
- Test: `tests/test_control.py`

**Interfaces:**
- Produces: `ControlWindow(config, on_start, on_stop, on_add_region, on_adjust_region, on_delete_region, on_apply, status_provider, monitor: Rect, on_close=None)`; `refresh_regions() -> None`; `selected_region() -> int | None`; attributes `region_list: QListWidget`, `add_button`, `adjust_button`, `delete_button`, `start_button`, `show_outlines: QCheckBox`.
- Row text format: `f"{index + 1}  ·  {rect.w} × {rect.h}"`, appended `"  (inactive)"` when the mode hides the row.

- [ ] **Step 1: Write the failing tests**

```python
def make_window(config, calls=None, monitor=None):
    calls = calls if calls is not None else []
    return ControlWindow(
        config,
        lambda: calls.append("start"),
        lambda: calls.append("stop"),
        lambda: calls.append("add"),
        lambda index: calls.append(("adjust", index)),
        lambda index: calls.append(("delete", index)),
        lambda cfg: calls.append("apply"),
        dict,
        monitor or Rect(0, 0, 1000, 1000),
    )


def test_rows_follow_the_saved_regions(qapp):
    cfg = Config(region_mode="multi", regions=[
        {"monitor_id": "0", "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.1},
        {"monitor_id": "0", "x": 0.0, "y": 0.5, "w": 0.2, "h": 0.1},
    ])
    window = make_window(cfg)
    assert window.region_list.count() == 2
    assert window.region_list.item(0).text() == "1  ·  500 × 100"
    window.close()


def test_selecting_a_row_enables_adjust_and_delete(qapp):
    cfg = Config(region_mode="multi", regions=[
        {"monitor_id": "0", "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.1},
        {"monitor_id": "0", "x": 0.0, "y": 0.5, "w": 0.2, "h": 0.1},
    ])
    window = make_window(cfg)
    window.region_list.setCurrentRow(1)
    assert window.selected_region() == 1
    assert window.adjust_button.isEnabled() and window.delete_button.isEnabled()
    window.close()


def test_delete_passes_the_selected_index(qapp):
    calls: list = []
    cfg = Config(region_mode="multi", regions=[
        {"monitor_id": "0", "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.1},
        {"monitor_id": "0", "x": 0.0, "y": 0.5, "w": 0.2, "h": 0.1},
    ])
    window = make_window(cfg, calls)
    window.region_list.setCurrentRow(1)
    window.delete_button.click()
    assert ("delete", 1) in calls
    window.close()


def test_delete_is_disabled_with_one_region(qapp):
    cfg = Config(regions=[{"monitor_id": "0", "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.1}])
    window = make_window(cfg)
    window.region_list.setCurrentRow(0)
    assert window.delete_button.isEnabled() is False
    window.close()


def test_start_is_blocked_without_regions(qapp):
    calls: list = []
    window = make_window(Config(regions=[]), calls)
    assert window.start_button.isEnabled() is False
    window.start_button.click()
    assert calls == []
    window.close()


def test_single_mode_marks_extra_rows_inactive_without_deleting_them(qapp):
    cfg = Config(region_mode="single", regions=[
        {"monitor_id": "0", "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.1},
        {"monitor_id": "0", "x": 0.0, "y": 0.5, "w": 0.2, "h": 0.1},
    ])
    window = make_window(cfg)
    assert window.region_list.count() == 2
    assert "inactive" in window.region_list.item(1).text()
    window.close()
```

Update the existing `make_window` helper in the file to the new signature (it is module-level, so the old one is replaced). Give the existing `test_start_button_calls_callback` a config with one region, because Start now requires one.

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_control.py -q`
Expected: FAIL — unexpected keyword `monitor`.

- [ ] **Step 3: Implement**

In `ui/control.py`:
- ctor takes `monitor: Rect` and the three region callbacks; store them.
- Add `self.region_list = QListWidget()`, `self.add_button`, `self.adjust_button`, `self.delete_button`, wired to the callbacks with `self.selected_region()`.
- `refresh_regions()` rebuilds rows from `self.config.regions` via `to_physical`, disables Delete when `len(regions) <= 1`, enables Adjust/Delete only with a selection, keeps the previous selection when possible, and enables `start_button` only when at least one region exists (status text points at Add when none).
- Call `refresh_regions()` in the ctor and when the mode combo changes.
- `app.py`: pass `RegionSet.from_config(cfg, screen)`'s monitor (`screen`) and callbacks that append `("add"/"adjust"/"delete")` to `actions` so Task 6 can wire them `None`-safe: `lambda: None`, `lambda index: None`, `lambda index: None` for now.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_control.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/ui/control.py src/ocrtrans/app.py tests/test_control.py
git commit -m "feat: region list with add, adjust and delete in the control window"
```

---

### Task 3: On-screen region outline

**Files:**
- Create: `src/ocrtrans/ui/outline.py`
- Modify: `src/ocrtrans/exclusion.py` (add `exclude_from_capture`)
- Modify: `src/ocrtrans/overlay.py:67-76` (import the moved helper)
- Modify: `src/ocrtrans/app.py` (create the outline; show/hide on stop/start)
- Test: `tests/test_outline.py`

**Interfaces:**
- Produces: `exclusion.exclude_from_capture(hwnd: int) -> bool`.
- Produces: `RegionOutline(QWidget)` with `__init__(screen: Rect)`, `set_regions(rects: list[Rect], active: int | None, inactive: set[int])`, `set_visible_for(stopped: bool, force: bool)`, and attributes `excluded: bool`, `rects`, `active`.

- [ ] **Step 1: Write the failing tests**

```python
from ocrtrans.geometry import Rect
from ocrtrans.ui import outline as outline_mod
from ocrtrans.ui.outline import RegionOutline


def test_outline_keeps_the_regions_and_selection(qapp):
    view = RegionOutline(Rect(0, 0, 1000, 1000))
    view.set_regions([Rect(10, 10, 100, 40), Rect(10, 60, 80, 20)], active=1, inactive=set())
    assert view.rects == [Rect(10, 10, 100, 40), Rect(10, 60, 80, 20)]
    assert view.active == 1
    view.close()


def test_outline_stays_hidden_while_running(qapp):
    view = RegionOutline(Rect(0, 0, 1000, 1000))
    assert view.set_visible_for(stopped=False, force=False) is False
    assert view.set_visible_for(stopped=True, force=False) is True
    assert view.set_visible_for(stopped=False, force=True) is True
    view.close()


def test_outline_refuses_to_show_when_exclusion_fails(qapp, monkeypatch):
    monkeypatch.setattr(outline_mod, "exclude_from_capture", lambda hwnd: False)
    view = RegionOutline(Rect(0, 0, 1000, 1000))
    view.set_regions([Rect(10, 10, 100, 40)], active=0, inactive=set())
    assert view.set_visible_for(stopped=True, force=False) is False
    assert view.excluded is False
    view.close()
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_outline.py -q`
Expected: FAIL — module `ocrtrans.ui.outline` not found.

- [ ] **Step 3: Implement**

- `exclusion.py`: move the ctypes body of `overlay._exclude_from_capture` to `exclude_from_capture(hwnd: int) -> bool` (same styles, same `WDA_EXCLUDEFROMCAPTURE`); `overlay.py` imports it and keeps its existing call site.
- `ui/outline.py`: frameless, topmost, `Tool`, `WA_TransparentForMouseEvents`, `WA_TranslucentBackground`; `paintEvent` draws a 2 px accent rect + a number badge per region, dashed `muted` for inactive indices, 3 px for the active one; `set_visible_for` returns the resulting visibility and only shows when `excluded` is true.
- `app.py`: construct `RegionOutline(screen)` next to the overlay; on `stop()` call `outline.set_visible_for(stopped=True, force=outline_force)` and `refresh_regions`; on `start()` call `set_visible_for(stopped=False, force=outline_force)`. Add a session-only "Show outlines while running" checkbox to the control window and read its value in `app.py`.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_outline.py tests/test_overlay_cover.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/exclusion.py src/ocrtrans/overlay.py src/ocrtrans/ui/outline.py src/ocrtrans/app.py src/ocrtrans/ui/control.py tests/test_outline.py
git commit -m "feat: draw the selected capture regions on screen"
```

---

### Task 4: Handle editor for one region

**Files:**
- Modify: `src/ocrtrans/regions.py` (pure drag math)
- Modify: `src/ocrtrans/ui/picker.py` (adjust mode)
- Test: `tests/test_regions.py`

**Interfaces:**
- Produces: `handle_at(rect: Rect, point: tuple[int, int], tolerance: int = 6) -> str | None` returning `"n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw" | "body" | None`.
- Produces: `drag_rect(rect: Rect, handle: str, dx: int, dy: int, monitor: Rect, min_size: int = 16) -> Rect`.
- Produces: `RegionPicker.adjust(rect: Rect) -> Rect | None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_handle_at_finds_each_edge_and_the_body():
    rect = Rect(100, 100, 200, 100)
    assert handle_at(rect, (100, 150)) == "w"
    assert handle_at(rect, (300, 150)) == "e"
    assert handle_at(rect, (200, 100)) == "n"
    assert handle_at(rect, (200, 200)) == "s"
    assert handle_at(rect, (100, 100)) == "nw"
    assert handle_at(rect, (300, 200)) == "se"
    assert handle_at(rect, (200, 150)) == "body"
    assert handle_at(rect, (50, 50)) is None


def test_drag_body_moves_the_rect():
    assert drag_rect(Rect(100, 100, 200, 100), "body", 10, 5, Rect(0, 0, 1000, 1000)) == Rect(110, 105, 200, 100)


def test_drag_east_edge_resizes_not_moves():
    assert drag_rect(Rect(100, 100, 200, 100), "e", 40, 0, Rect(0, 0, 1000, 1000)) == Rect(100, 100, 240, 100)


def test_drag_past_the_monitor_clamps():
    assert drag_rect(Rect(100, 100, 200, 100), "w", -500, 0, Rect(0, 0, 1000, 1000)) == Rect(0, 100, 300, 100)


def test_drag_below_the_minimum_keeps_the_minimum():
    assert drag_rect(Rect(100, 100, 200, 100), "e", -500, 0, Rect(0, 0, 1000, 1000)).w == 16
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_regions.py -q`
Expected: FAIL — names not defined.

- [ ] **Step 3: Implement**

- `regions.py`: `handle_at` (tolerance box on each corner first, then edges, then inside), `drag_rect` (adjust the named edges, enforce `min_size`, clamp with `clamp_rect`).
- `picker.py`: `RegionPicker.adjust(rect) -> Rect | None`; cursor changes over a handle; `mousePress` stores handle + origin; `mouseMove` calls `drag_rect` and repaints; `mouseRelease` accepts and closes; `Esc` closes with `None`. Keep `pick()` unchanged.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_regions.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/regions.py src/ocrtrans/ui/picker.py tests/test_regions.py
git commit -m "feat: drag handles to move and resize a capture region"
```

---

### Task 5: Dark theme, appearance panel, live preview

**Files:**
- Create: `src/ocrtrans/ui/theme.py`
- Modify: `src/ocrtrans/ui/control.py` (appearance group + preview widget)
- Modify: `src/ocrtrans/overlay.py:40-65` (rebuild `QtMeasure` when the family changes)
- Modify: `src/ocrtrans/app.py` (call `theme.apply`)
- Test: `tests/test_control.py`, `tests/test_overlay_cover.py`

**Interfaces:**
- Produces: `theme.QSS: str`, `theme.apply(app: QApplication) -> None`, and the color constants `BG, PANEL, BORDER, TEXT, MUTED, ACCENT, DANGER`.
- Produces in `ControlWindow`: `text_color_button`, `bg_color_button`, `font_combo`, `preview`; `read_config()` now persists `text_color`, `bg_color`, `font_family`, `min_font_px`.

- [ ] **Step 1: Write the failing tests**

```python
def test_read_config_persists_the_appearance(qapp):
    window = make_window(Config())
    window.set_appearance(text_color="#FFE08A", bg_color="#000000", font_family="Consolas", min_font_px=14)
    cfg = window.read_config()
    assert (cfg.text_color, cfg.bg_color, cfg.font_family, cfg.min_font_px) == (
        "#FFE08A", "#000000", "Consolas", 14)
    window.close()


def test_theme_is_applied_once(qapp):
    theme.apply(qapp)
    assert theme.ACCENT in qapp.styleSheet()
```

In `tests/test_overlay_cover.py`:

```python
def test_overlay_picks_up_a_font_change(qapp):
    cfg = Config()
    view = OverlayWindow(Rect(0, 0, 800, 200), cfg, queue.Queue())
    view.show_items([RenderItem(1, "xin chào", Rect(0, 0, 400, 60), "hello", False, 20)])
    cfg.font_family = "Consolas"
    view.show_items([RenderItem(1, "xin chào", Rect(0, 0, 400, 60), "hello", False, 20)])
    assert view.measure.family == "Consolas"
    view.close()
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_control.py tests/test_overlay_cover.py -q`
Expected: FAIL — `set_appearance`, `theme` missing; second test fails on the family.

- [ ] **Step 3: Implement**

- `theme.py`: the token constants and a `QSS` string styling `QWidget`, `QListWidget` (selected row accent), `QPushButton`, `QLineEdit`, `QComboBox`, `QSpinBox`, `QCheckBox`, `QLabel#muted`, `QPushButton#danger`; `apply(app)` sets `app.setStyleSheet(QSS)` and the dark `QPalette`.
- `control.py`: add the Overlay group (two swatch buttons opening `QColorDialog`, `QFontComboBox`, the min-font spin moved there), a `Preview` widget painting a Vietnamese sample with the current values, and `set_appearance(text_color, bg_color, font_family, min_font_px)`. Restructure the window per the spec (header, regions panel, overlay panel, footer) and use `.label` styling for section titles.
- `overlay.py`: in `_recompute`, `if self.measure.family != self.config.font_family: self.measure = QtMeasure(self.config.font_family, self.dpr)`.
- `app.py`: `theme.apply(app)` after `QApplication(sys.argv)`.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_control.py tests/test_overlay_cover.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/ui/theme.py src/ocrtrans/ui/control.py src/ocrtrans/overlay.py src/ocrtrans/app.py tests/test_control.py tests/test_overlay_cover.py
git commit -m "feat: dark control window with overlay appearance settings"
```

---

### Task 6: Wire regions end to end

**Files:**
- Modify: `src/ocrtrans/regions.py` (list operations)
- Modify: `src/ocrtrans/app.py:239-255`, `:286-301`
- Modify: `README.md` (regions section)
- Test: `tests/test_regions.py`

**Interfaces:**
- Produces: `add_region(regions: list[NormalizedRegion], rect: Rect, monitor: Rect) -> list[NormalizedRegion]`, `replace_region(regions, index: int, rect: Rect, monitor: Rect) -> list[NormalizedRegion]`, `remove_region(regions, index: int) -> list[NormalizedRegion]` (never returns an empty list).

- [ ] **Step 1: Write the failing tests**

```python
def test_add_region_appends_a_normalized_rect():
    monitor = Rect(0, 0, 1000, 1000)
    out = add_region([], Rect(100, 200, 300, 100), monitor)
    assert len(out) == 1 and out[0].x == 0.1 and out[0].w == 0.3


def test_replace_region_keeps_the_order():
    monitor = Rect(0, 0, 1000, 1000)
    regions = add_region(add_region([], Rect(0, 0, 100, 50), monitor), Rect(0, 500, 100, 50), monitor)
    out = replace_region(regions, 1, Rect(0, 600, 200, 50), monitor)
    assert out[0] == regions[0] and out[1].y == 0.6 and out[1].w == 0.2


def test_remove_region_never_empties_the_list():
    monitor = Rect(0, 0, 1000, 1000)
    one = add_region([], Rect(0, 0, 100, 50), monitor)
    assert remove_region(one, 0) == one
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python -m pytest tests/test_regions.py -q`
Expected: FAIL — names not defined.

- [ ] **Step 3: Implement**

- `regions.py`: the three functions, all clamping before normalizing.
- `app.py`: `pick()` becomes `add_region()` (hotkey included); new `adjust_region(index)` opens `RegionPicker.adjust` for `to_physical(cfg.regions[index])`; new `delete_region(index)`; each mutates `cfg.regions`, calls `save_config`, pushes the new rects into `pipeline.set_regions` + `overlay.set_regions`, then `window.refresh_regions()` + `outline.set_regions(...)`; `start()` refuses with `state["error"] = "add a region first"` when the list is empty, and hides the outline.
- `README.md`: describe the list, Add/Adjust/Delete, the outline and the appearance settings; remove the `monitor` mode.

- [ ] **Step 4: Run the tests and the smoke**

Run: `.venv/Scripts/python -m pytest -q`
Expected: PASS.
Run: `.venv/Scripts/python -c "import sys; from ocrtrans.app import main; sys.argv=['ocrtrans','--smoke']; raise SystemExit(main())"`
Expected: the banner shows `WgcBackend`, no traceback, and `stats:` is printed.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/regions.py src/ocrtrans/app.py README.md tests/test_regions.py
git commit -m "feat: add, adjust and delete capture regions from the window"
```

---

## Self-review notes

- Every spec section maps to a task: modes/migration (1), list (2), outline (3), adjust (4), look and appearance (5), wiring and docs (6).
- The spec's "no opacity setting" needs no task; it is a constraint, and no task adds one.
- `RegionOutline.set_visible_for` is the only visibility switch, so Review Focus 4 has one owner.
