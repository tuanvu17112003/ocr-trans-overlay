# Region manager and overlay settings — design

Date: 2026-10-24
Status: approved approach C, awaiting design sign-off

## Intent

The app currently exposes one "Pick region" button and a label that says
"1 region(s)". Nothing shows *which* rectangle will be captured, so a wrong
or stale region is invisible until the overlay fails. Multi-region exists in
the data model but the picker always replaces the list with a single rect, so
the mode is unusable.

This version removes the whole-monitor mode (too much screen to scan, and it
made the overlay window full-screen), makes regions a visible, editable list,
and exposes the overlay appearance settings that already exist in `Config`.

## Goals

- Remove the `monitor` region mode. Old configs migrate: keep the saved
  regions; if none, the whole screen becomes one region.
- A region list in the control window: index, size, and the rectangle drawn on
  screen. Selecting a row highlights that region.
- Add / Adjust / Delete a region from that list.
- Adjust mode: the selected rectangle gets 8 handles; drag the body to move,
  drag a handle to resize; Enter accepts, Esc cancels.
- Overlay settings in the window: text color, background color, font family,
  minimum font size, with a live preview inside the window.
- No opacity setting: the cover must stay opaque so the source text is hidden.

## Non-goals

- Dragging several regions at once, snapping, or guide lines.
- Per-region colors, per-region fonts, or per-region translation settings.
- Reordering regions (order does not affect behaviour).
- Changing OCR, translation, capture or overlay rendering logic.

## Anti-goals

- Showing the source text through a translucent cover.
- A full-screen interactive editor that stays on screen while the app runs.
- Any change that makes the running pipeline slower.

## Constraints

- PyQt6, Windows only, no process injection, no global mouse hooks.
- Every region is stored normalized to the monitor, as today
  (`NormalizedRegion`, monitor id "0").
- The outline window must be excluded from capture
  (`WDA_EXCLUDEFROMCAPTURE`), otherwise OCR reads the outlines.
- Capture defaults to WGC; that does not change here.
- Tests run offline; Qt tests use the existing offscreen setup.

## Approaches considered

- **A. List plus on-screen outline.** Cheap, solves "which region", but the
  rectangle can only be replaced, never moved.
- **B. Full on-screen editor.** Everything draggable, no list. Most direct,
  but needs hit-testing, keys and a persistent editor over the video.
- **C. A plus a temporary adjust editor (chosen).** List manages, the selected
  rect opens in a temporary handle editor. Keeps the running path untouched.

C wins because the list is what tells the user what exists, and the editor is
modal and short-lived, so it cannot interfere with the overlay or hotkeys.

## Design

### 1. Config and migration

- `REGION_MODES = {"single", "multi"}`.
- `load_config`: if the file says `region_mode == "monitor"`, load it as
  `single` and keep `regions` untouched.
- `RegionSet.from_config`: delete the `monitor` branch. Empty `regions` still
  falls back to the full screen as one region (unchanged).
- `app.MODES` and the mode combo become `["single", "multi"]`; the hotkey
  cycle follows.

### 2. Region list

`ControlWindow` gets a regions group:

- `QListWidget` rows: `"1  —  1505 x 373"`. Row *n* is region *n*.
- Selecting a row sets the outline selection and enables Adjust/Delete.
- Buttons: **Add** (drag a new rect on screen, appended), **Adjust** (opens
  the editor for the selected row), **Delete** (removes the selected row;
  disabled when only one remains).
- In `single` mode only row 1 is applied; other rows stay in the config and
  are shown greyed with a "single mode" hint, so switching back to `multi`
  restores them. The outline draws applied regions solid and inactive rows
  dashed, both numbered, so the list and the screen agree.
- Selecting a row always selects that row, in either mode; editing a greyed
  row is allowed and does not change the mode.
- Changing the list calls the existing `apply_regions(list[Rect])` path:
  `save_config`, `pipeline.set_regions`, `overlay.set_regions`, refresh the
  outline.

### 3. Outline window

A small `RegionOutline(QWidget)` in `ui/outline.py`:

- Frameless, topmost, tool window, click-through, translucent background,
  `WDA_EXCLUDEFROMCAPTURE`. If the exclusion call fails, do not show it
  (log once) — a captured outline would poison OCR.
- One full-screen window that draws each region as a 2 px rectangle plus a
  numbered badge; the selected region uses the accent color and a thicker
  border. Inactive rows in `single` mode are dashed.
- Geometry follows `RegionSet.from_config`, refreshed on every list change,
  on mode change, and when the control window is shown.
- Shown while the control window is visible **and the pipeline is stopped**.
  It hides automatically on Start, because a full-screen layered window over a
  playing video is exactly what `overlay.py` avoids. A session-only
  "Show outlines while running" checkbox can force it back on.

### 4. Adjust editor

Extend `ui/picker.py`:

- `handle_at(rect, point, tolerance) -> str | None` as a pure function in
  `regions.py` (no Qt) returning `n/s/e/w/ne/nw/se/sw/body`.
- `RegionPicker.adjust(rect) -> Rect | None`: show the rect with 8 square
  handles (12 px) and a shaded body; drag body moves, drag handle resizes,
  keep the rect inside the monitor and at least 16x16; Enter or double-click
  accepts, Esc cancels (`None` = keep the old rect).
- `pick()` keeps its current behaviour for Add. The `region` hotkey now
  appends a region (same as Add) instead of replacing the whole list.

### 5. Appearance settings

New group in the control window:

- Text color and background color: swatch buttons opening `QColorDialog`.
- Font: `QFontComboBox`.
- Minimum font px: existing spin box moves here.
- Preview: a fixed-height widget painting a Vietnamese sample line with the
  chosen font, size and both colors, updated on every change.
- Every change writes the shared `Config` object and `save_config`; the
  `OverlayWindow` reads `config` at paint/layout time, so a running overlay
  picks the change up on the next render without a restart.

### 6. Edge cases

- Deleting the last region is disabled; `single` with an empty list falls back
  to the full screen (existing behaviour).
- A stored region that no longer fits the monitor is clamped when it is
  resolved (`RegionSet.from_config`), not when it is loaded, and the clamped
  value is **not** written back: the original geometry survives a monitor swap.
- Adjust/pick are modal to their own window; the control window cannot issue
  two at once (button disabled while active).
- If the app is running, list changes go through `set_regions`, which already
  creates a new epoch and clears the overlay.

### 7. Known limitation (out of scope)

Coordinates are Qt logical pixels today (`primaryScreen().geometry()`), while
capture crops the physical frame. At 100% scaling they are identical, and the
target machine measures `dpr = 1.0`; the outline therefore mirrors exactly the
rects the pipeline crops. On a HiDPI display the existing picker, regions and
capture are all off by the scale factor — a separate fix, not part of this
design.

## Tests

- `Config`: `monitor` config loads as `single`; unknown mode still rejected.
- `RegionSet`: single returns one, multi returns all, empty returns the screen.
- `regions`: `handle_at` for body, each corner/edge and outside; clamping.
- `ControlWindow` (offscreen): add/adjust/delete update the config list; mode
  switch keeps extra regions; delete is disabled at one.
- `RegionOutline`: geometry and selected index from a region list.

## Left to the builder

- Exact widget layout, spacing and icons; badge position and colors.
- Handle hit tolerance (6 px) and badge size.
- Keyboard accelerators beyond Enter/Esc.
- Whether to animate the outline when the selection changes.

## Calls made (check these)

- Outlines are shown only while the pipeline is stopped (plus a session-only
  "show while running" checkbox); nothing is added to the config for it.
- Single mode keeps extra regions in the config instead of deleting them, and
  draws them dashed.
- Delete is disabled at one region rather than allowing an empty list.
- Adjust and Add are separate buttons (no click-drag on existing outlines);
  the region hotkey appends, it never replaces.
- Regions are clamped when resolved, never written back to the config.
- HiDPI correctness is out of scope (measured dpr 1.0 here).
