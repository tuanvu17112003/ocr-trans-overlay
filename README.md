# OCRTrans — realtime screen translation overlay (Windows)

Captures a game region, OCRs English text, translates it to Vietnamese and paints the
translation over the original text in a click-through overlay that screen capture does
not see. Local engines by default; free online engines are opt-in.

## Requirements

- Windows 11 (tested on 25H2, build 26200)
- Python 3.13
- NVIDIA GPU with CUDA 13 driver (tested on RTX 3080 10 GB)

## Install

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.14.1+cu130` from the PyTorch CUDA index and
`transformers<5` (5.x breaks the EnViT5 tokenizer and weight tying).

## Models

Downloaded once into the Hugging Face cache on first start:

| Engine | Repo | Size on disk |
|---|---|---|
| `nllb` (default) | `facebook/nllb-200-distilled-600M` | ~2.4 GB |
| `envit5` | `VietAI/envit5-translation` | ~1.0 GB |

OCR models ship inside the `rapidocr` wheel (PP-OCRv6 det/rec small).

## Run

```bat
run.bat
```

or `.venv\Scripts\python -m ocrtrans.app`.

## Hotkeys

| Keys | Action |
|---|---|
| `Ctrl+Alt+Z` | pause / resume translation |
| `Ctrl+Alt+R` | drag a new capture region |
| `Ctrl+Alt+M` | cycle region mode (single / multi / whole monitor) |
| `Ctrl+Alt+H` | toggle the performance HUD |
| `Ctrl+Alt+S` | settings |
| `Ctrl+Alt+Q` | quit |

Settings and regions are stored in `%LOCALAPPDATA%\OCRTrans\config.json`; logs in
`%LOCALAPPDATA%\OCRTrans\app.log`.

## Engines

| Engine | Kind | Free tier | Notes |
|---|---|---|---|
| NLLB-600M | local | unlimited | ~1.25 GB VRAM, best default |
| EnViT5 | local | unlimited | ~0.64 GB VRAM, fastest local |
| Google (unofficial) | online | unofficial, can throttle/block | no key, may break without notice |
| Gemini | online | free API quota (per-project, changes over time) | needs API key |
| DeepL Free | online | 500k chars/month, 1 request/s | needs API key |
| MyMemory | online | anonymous quota per day | no key, short texts |

Online engines are off by default and require the consent checkbox in settings. If a
provider returns 429/401 the router cools it down for 60 s and uses NLLB when
`fallback_to_local` is on.

## Measured (RTX 3080, 2560x1440 HDR, replay harness)

| Engine | OCR p50 | MT p50 | end-to-end p50 | p95 |
|---|---|---|---|---|
| nllb | 180 ms | 31 ms | 214 ms | 717 ms |
| envit5 | 174 ms | 59 ms | 237 ms | 424 ms |

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

## Tests

```bat
.venv\Scripts\python -m pytest -q          :: unit tests (GPU tests skipped)
.venv\Scripts\python -m pytest -m gpu -q   :: CUDA model tests
```
