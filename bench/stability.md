# Stability measurements

Everything below was measured on this machine (RTX 3080, Windows 11 25H2) with
`scripts/stability_bench.py`, which drives the real pipeline with a fake capture
and synthetic frames so the numbers are comparable run to run.

Re-run it with:

```bat
.venv\Scripts\python.exe scripts\stability_bench.py
```

The "before" column is the build at tag `v0.1.0` (commit 354fe53). The "after"
column is the current branch.

## Scan cadence and cover stability

Synthetic caption rendered on a random-noise background; the background frame
changes on every grab, which is what makes the recogniser wobble.

| scenario | metric | before | after |
|---|---|---|---|
| static caption, static background | pass/s | 2.00 | 2.00 |
| static caption, **moving background** | pass/s | **3.96** | **1.50** |
| static caption, moving background | cover on/off transitions | **3** | **1** |
| static caption, moving background | translation requests | **3** | **1** |

Before: one character of OCR noise ("traveler." -> "traveler") reset the steady
cadence and tore the cover down, so the overlay visibly strobed while the text on
screen never changed. After: the cadence is keyed on
`textnorm.scan_fingerprint` compared with a similarity threshold, and
`OverlayState` holds the cover through a cosmetic wobble instead of dropping it.

The 1.50 pass/s figure is dominated by the first pass, which includes the one-off
model load (~2.1 s on the CPU provider on this machine); in a steady state the
decay settles at one pass per second.

## Artwork rejection

Synthetic 4x3 icon grid (rounded tiles, ellipse glyphs, bar underline) rendered
on a dark background, and a real caption, both through RapidOCR with the same
detector settings the app uses.

| scene | raw detector boxes | kept by the gate | translated |
|---|---|---|---|
| icon grid, **no text at all** | 8 | **0** | **0** (was 4) |
| real caption | 1 | 1 | 1 |

Rejection reasons on the icon grid: `content` x8 (too few alphanumerics, aspect
outside the band). Raising `Det.box_thresh` from 0.3 to PaddleOCR's default 0.6
did **not** change the raw count in earlier probes, so the fix had to be a filter
after recognition rather than a threshold change.

## What was measured and rejected

Two ideas looked reasonable and did not survive measurement, so they are not in
the code:

| idea | measurement | verdict |
|---|---|---|
| stroke-width band as a text/artwork discriminator | rendered text 0.095-0.239; icon grid 0.077-0.086; wood grain 0.100-0.149; white noise 0.046-0.064 | bands overlap completely; `candidates.stroke_ratio` kept as a diagnostic only |
| pixel-diff dirty region to skip OCR | with video under the glyphs the "text mask" changes on 100 % of frames (dirty ratio 0.22) | cannot gate scanning; only a text-shape mask could, and that was out of scope for this pass |

## Cover damage area

| scenario | before | after |
|---|---|---|
| 1 of 8 blocks changes | 904x706 px damage = **42 % of the overlay window** | 904x48 px + inset, **one block** |
| rect/line-height jitter of 1-2 px | **12 of 12 scans repainted** | 0 repaints (`SNAP_PX=3` tolerance) |
| block A when block B is translated | font 14 -> 12 px, cover 42 -> 36 px | unchanged |

## Test suite

`pytest -q`: 264 passed, 5 skipped, 3 deselected (~49 s).
The real-game suite (`-m real --run-real`) still needs the game open; nothing in
this pass touched it.
