# OCRTrans — realtime screen translation overlay (Windows)

Captures a game region, OCRs English text, translates it to Vietnamese and paints the
translation over the original text in a click-through overlay that screen capture does
not see. Local engines by default; free online engines are opt-in.

## Requirements

- Windows 11 (tested on 25H2, build 26200)
- Python 3.13
- NVIDIA GPU with CUDA 13 driver (tested on RTX 3080 10 GB)

## Install

```powershell
.\install.ps1          # NVIDIA GPU, CUDA build (NLLB + Argos), ~3 GB of wheels
.\install.ps1 -Cpu     # any machine, CPU wheels (~400 MB), Argos + online engines
```

The script creates `.venv` with `py -3.13`, installs the pinned requirements and the
package itself (`pip install -e . --no-deps`). `-DryRun` prints the commands without
running them. Requirements are pinned; `requirements.txt` pulls `torch==2.14.1+cu130`
from the PyTorch CUDA index, `requirements-cpu.txt` lets `stanza` bring the CPU wheel.

The CPU build has no `transformers`, so NLLB cannot load there: the app falls back to
**Argos** and says so on the status line. Online engines work if you allow them.

Manual install, if you prefer:

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

## Models

Downloaded once into the Hugging Face cache on first start:

| Engine | Repo | Size on disk |
|---|---|---|
| `nllb` (default) | `facebook/nllb-200-distilled-600M` | ~2.4 GB |
| `envit5` | `VietAI/envit5-translation` | ~1.0 GB |
| `argos` (optional) | Argos en->vi package, downloaded on first use | ~0.8 GB |

OCR models ship inside the `rapidocr` wheel (PP-OCRv6 det/rec small).

`argos` needs the optional package, installed without dependencies so the CUDA
`onnxruntime-gpu` is not replaced by the CPU build:

```bat
.venv\Scripts\python -m pip install --no-deps argostranslate ctranslate2 "stanza==1.10.1" minisbd emoji
```

It runs on CPU and downloads its language packs into `%USERPROFILE%\.local\share\argos-translate`
on first start (a few hundred MB; the en->vi route may pivot).

## Run

```bat
run.bat
```

or `.venv\Scripts\python -m ocrtrans.app`. The control window opens with every setting:
engine, keys, stability profile, fps, the region list, overlay colors/font and
Start/Stop plus live status (engine, capture health, OCR passes and scan rate, shown/tiled,
p50/p95).
Nothing is downloaded or captured until Start. Closing the window quits.

**Regions.** The window lists every saved region with its size. `Add region` drags a new
one (it is appended, never replaced), `Adjust` opens the selected rectangle with eight
handles (drag the body to move, a handle to resize, `Enter` applies, `Esc` cancels) and
`Delete` removes it (disabled when one is left). Selected regions are outlined on screen
while translation is stopped; a full-screen outline would make the desktop compositor
recomposite the video, so the outline only covers the rectangles and hides while running
unless `Show outlines while running` is ticked. In `single` mode only the first region is
captured; the others stay saved and show as inactive.

**Overlay.** Text color, cover color (opaque, it must hide the source text), font family
and minimum font size are set in the window with a live preview; there is no opacity
setting on purpose.

Debug switches (environment variables):

| Variable | Effect |
|---|---|
| `OCRTRANS_DEBUG=1` | debug logging from the first second, same as ticking `Debug logging` in the window |
| `OCRTRANS_NO_OVERLAY=1` | run capture/OCR/MT without showing the overlay (A/B the overlay) |
| `OCRTRANS_DXGI=1` | force Desktop Duplication instead of the default Windows Graphics Capture (A/B the capture path) |

The log lives at `%LOCALAPPDATA%\OCRTrans\app.log`. At INFO it prints a `stats {...}` line
every 5 s with `grab_p50/p95`, `ocr_p50/p95`, translation `p50/p95`, `scan_rate`,
`blocks`, `rendered`; any scan over 400 ms or translation over 500 ms is logged as a
warning with its line/char count.

## Hotkeys

| Keys | Action |
|---|---|
| `Ctrl+Alt+Z` | start / stop translation |
| `Ctrl+Alt+R` | add a capture region (drag a new rectangle) |
| `Ctrl+Alt+M` | cycle region mode (single / multi) |
| `Ctrl+Alt+H` | toggle the performance HUD |
| `Ctrl+Alt+S` | raise the control window |
| `Ctrl+Alt+Q` | quit |

Settings and regions are stored in `%LOCALAPPDATA%\OCRTrans\config.json`; logs in
`%LOCALAPPDATA%\OCRTrans\app.log`.

## Engines

| Engine | Kind | Free tier | Notes |
|---|---|---|---|
| NLLB-600M | local | unlimited | ~1.25 GB VRAM, best default |
| EnViT5 | local | unlimited | ~0.64 GB VRAM, fastest local |
| Argos | local | unlimited | CPU only, small model, optional install |
| Google (unofficial) | online | unofficial, can throttle/block | no key, may break without notice |
| Gemini | online | free API quota (per-project, changes over time) | needs API key |
| DeepL Free | online | 500k chars/month, 1 request/s | needs API key |
| MyMemory | online | anonymous quota per day | no key, short texts |

Online engines are off by default and require the consent checkbox in settings. If a
provider returns 429/401 the router cools it down for 60 s and uses NLLB when
`fallback_to_local` is on.

## Measured (RTX 3080, 2560x1440 HDR)

Replay harness:

| Engine | OCR p50 | MT p50 | end-to-end p50 | p95 |
|---|---|---|---|---|
| nllb | 211 ms | 46 ms | 263 ms | 835 ms |
| envit5 | 210 ms | 72 ms | 286 ms | 500 ms |

Real Gunfire Reborn window (WGC, settings screen, 15 s): OCR p50 265 ms / p95 404 ms,
translation p50 126 ms / p95 328 ms, 11 OCR passes, no queue backlog.

OCR over 30 rendered lines: mean text similarity 0.961, exact 0.767. Numbers preserved
98.75%, proper nouns 88.75% (NLLB) / 87.5% (EnViT5) on the 80-line corpus.

Run the harness yourself:

```bat
.venv\Scripts\python -m ocrtrans.bench --engine all --quality --perf --ocr --no-comet
```

## Known limits

- **HDR**: the display is HDR, so DXGI returns `rgba16f`. Capture always tonemaps through
  `to_bgr()`; a non-HDR screen works too.
- **Exclusive fullscreen** is not supported; use borderless windowed.
- **DRM-protected** windows and some anti-cheat overlays capture as black. The capture
  watchdog falls back from DXGI to WGC after 3 black frames or 2 access losses.
- **Anti-cheat**: no injection, no process memory reads, no global keyboard hooks. The
  overlay is a separate top-level window and hotkeys use `RegisterHotKey`. Some
  competitive games still disallow overlays; check their rules.
- **API keys are stored in plaintext** in `config.json`.
- A capture region that stays nearly black for 3 frames (letterbox, night scene, DRM) is
  treated as `BLACK`: the overlay freezes, then hides, and capture falls back to WGC.
- The overlay hides itself from capture with `WDA_EXCLUDEFROMCAPTURE`; the hide-and-ack
  fallback exists for machines where affinity is ignored, and trips after 3 timeouts.
- Region rectangles are stored as fractions of the monitor, so they survive resolution
  changes; a >2% aspect change is reported by `remap()`.
- Multi-monitor capture uses the union rectangle of the selected regions.
- The overlay covers the whole OCR block with an opaque background and draws the
  Vietnamese inside it: the English source is never visible, and a block stays hidden
  until its translation for the current text has arrived (no mismatched pairs).
- The overlay window is the size of the selected text region (plus 8 px), not the whole
  screen, and a change repaints only the changed block. A full-screen layered window
  makes the desktop compositor redo the whole video on every hide/show; the picture
  freezes while the audio keeps playing.
- Text is sized from the source line height and never painted outside the block: long
  Vietnamese shrinks to the font floor, then ends with an ellipsis (logged) instead of
  covering neighbouring text. Lines are broken by dynamic programming over the words at
  the current font size (least ragged fit, no one-word last line), so a block keeps the
  largest font that still fits its box instead of accepting the first greedy wrap.
- Detection runs with the long side capped at 1280 px and lower thresholds than the
  PaddleOCR defaults, which reads faint HUD text and roughly halves full-frame OCR time
  on a synthetic 2560x1440 scene (353 -> 171 ms, CPU onnxruntime). A large region whose long
  side exceeds 1600 px gets one bounded tiled rescue pass (max 2x2 tiles, 64 px overlap, at most
  once per 2 s) only when a single pass finds nothing. Tiles are skipped for wide, short
  strips (under 200 px tall): 20 px captions already survive the cap, and tiling a
  subtitle strip only tripled the OCR work at the moment of a miss.
- Paragraphs are merged up to 6 lines; two labels on the same row stay separate blocks.
- Scans are text-first: the full region is OCR'd at a fixed budget (one pass in flight,
  the next start is measured from the end of the previous pass and never faster than
  1.5x the last pass time), so background motion, video or a moving camera costs nothing.
  While the text set is unchanged it scans at 2 passes/s instead of 4, so a caption that
  sits on screen for seconds costs half as much. Capture grabs only when a pass is due
  (plus a 250 ms sample while empty), so the loop costs ~20% of one core instead of a busy
  core. A screen with no text backs off to one pass per second until a cheap 320 px luma
  diff says something changed. The status line shows `text N`, `shown N`, `tiled N`,
  `scan N ms` and `hidden N`.
- Black-screen detection samples the whole monitor, not the OCR strip: an empty or dark
  subtitle area is normal and no longer trips the WGC fallback. The overlay itself is
  excluded from capture, so it never feeds back into OCR.
- OCR logs its execution provider once at the first pass (`app.log`). If CUDA is not in
  use it says so and caps onnxruntime at 2 threads: the default grabs every core for
  ~80 ms per pass, which is enough to stall a video playing next to it.
- Exact text decides validity: a one-character change ("100 HP" -> "10 HP") hides the
  block at once, and a translation can only paint over the text it was requested for.
  A block that is no longer in the snapshot disappears; the 0.4-1.0 s TTL keep exists so
  one false-negative pass does not strobe the cover.
- Changing the engine while stopped frees the old model before the new one loads, so
  only one model sits in VRAM. Stop alone keeps the model loaded for a fast restart.
- NLLB-600M still mistranslates short game labels out of context. Paragraph grouping
  helps; it is not a substitute for a glossary or a stronger model.
- The overlay only paints a block once its exact text has been read for the stability
  window (fast 150 ms, safe 300 ms), so a typewriter dialogue appears after it settles,
  not while it is still animating.

## Privacy and security

- **No telemetry.** The app only connects to the network when an online engine is used.
- **Local engines** (`argos`, `nllb`) run on this machine; captured text never leaves it.
- **Online engines** (`google`, `gemini`, `deepl`, `mymemory`) send the recognised text to
  that provider, and only after `Allow captured text to reach an online provider` is ticked.
  The consent box resets per run.
- **API keys** are stored in plain text in `%LOCALAPPDATA%\OCRTrans\config.json`. The file
  stays on this machine; never copy it into a repository or a bug report.
- **Capture** uses Windows Graphics Capture (Desktop Duplication as a fallback). The overlay
  and the region outline set `WDA_EXCLUDEFROMCAPTURE`, so they do not show up in screen
  recordings of their own windows. Use this only on content you are allowed to view: apps
  protected by DRM or by an anti-cheat may refuse capture, and some consider capture
  exclusion against their terms.
- **Hotkeys** are registered with `RegisterHotKey`. The app does not inject into other
  processes, read their memory, or install global mouse hooks.

## Tests

```bat
.venv\Scripts\python -m pytest -q                     :: unit tests
.venv\Scripts\python -m pytest -m gpu -q              :: CUDA model tests
.venv\Scripts\python -m pytest -m real tests/real --run-real -q   :: real game, needs it open
```

The real suite captures the running Gunfire Reborn window, runs real OCR and NLLB, and
fails if a pipeline thread dies. Without `--run-real` (or with the game closed) it skips.

## License

MIT - see [LICENSE](LICENSE).
