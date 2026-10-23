# Cover-Mode Overlay Design

**Date:** 2026-10-24
**Status:** approved

## Problem

Observed on a translated web page: dark bars cover only part of each paragraph, English stays visible, and some blocks show a translation that belongs to different text.

Measured causes (synthetic paragraph, real OCR + fit):

1. Overlay paints the background at the fitted text rect. A short Vietnamese translation leaves the rest of the English block uncovered (119 px block, 48 px background).
2. With no translation yet, `_render` uses `text=block.translated or block.source`, so English is drawn inside the bar.
3. A changed source keeps the old translation for up to `stale_cap_s` (1 s).
4. `group_blocks` allows `gap >= -line_h`, so two labels on the same row merge into one block (`"Vietnamese English"`) and get one translation.
5. `max_lines=4` splits a five-line paragraph into 4 + 1, translating the last line without context.

## Decision

Cover mode: the overlay is opaque over the entire OCR block, and only Vietnamese is drawn.

## Rules

- Background rect = the source block rect (`item.rect`), always fully painted.
- Text rect = the block inset 2 px on each side, vertically centered, never outside the block.
- No translation for the current source text means no paint for that block.
- A changed source text clears the old translation on the same pass.
- Text comparisons fold case and whitespace so OCR jitter does not flicker blocks.
- `group_blocks` merges only when `gap >= -0.2 * line_height`, up to 6 lines.
- Font size still comes from one OCR line height, shrink-only, floor then ellipsis.

## Non-goals

- Per-word or per-line replacement, background matching the scene, or font-face imitation.
- Vertical-list detection (a list with tight spacing may still merge).

## Tests

- Background covers the block and is opaque; text stays inside and is centered.
- A changed source hides the old translation; a pending translation paints nothing.
- Same-row labels stay separate; a five-line paragraph stays one block.
- Full suite, GPU suite, real game suite, app smoke, plus a rendered proof image.
