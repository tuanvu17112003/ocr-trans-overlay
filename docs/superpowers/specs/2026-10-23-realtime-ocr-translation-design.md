# Realtime OCR Screen Translation Overlay (Windows) — Design

Date: 2026-10-23
Status: approved 2026-10-23

## Intent

Play English games without mentally translating. Capture on-screen English text,
OCR it, translate to Vietnamese, and paint the Vietnamese over the original text
in place. Single user, local machine, open source, no paid services.

Priority order, from the owner: latency first, translation quality second.

## Goals

- In-place Vietnamese overlay over the original English text region.
- Latency, measured from the pass where the tracker first sees the final text of a
  block to the frame where the Vietnamese pixel is on screen (includes the
  stability gate, excludes the game's own reveal animation). Two profiles:
  - `safe` gate (default, 134 ms): local uncached p50 <= 550 ms, P95 <= 700 ms,
    P99 <= 900 ms
  - `fast` gate (67 ms, premature translations replaced in place): local uncached
    p50 <= 300 ms, P95 <= 500 ms
  - local cached: p50 <= 200 ms
  - online Google/DeepL/MyMemory: p50 <= 700 ms, P95 <= 1200 ms
  - online Gemini: p50 <= 1500 ms, P95 <= 2600 ms, P99 <= 4000 ms
  The first version asked for 300 ms end-to-end; that is not reachable with a
  truthful typewriter gate on a 15 Hz capture loop, so the `safe` profile targets
  550 ms and the `fast` profile keeps 300 ms with a measured quality cost.
- Visible performance numbers so the owner can choose an engine on evidence:
  a one-shot 5-sentence benchmark per engine, plus a live HUD
  (capture ms, OCR ms, MT ms, end-to-end ms, updates/s, cache hits, p50/p95).
- Six selectable translation engines, switchable at runtime:
  - Local: `facebook/nllb-200-distilled-600M` (fast)
  - Local: `VietAI/envit5-translation` (better Vietnamese fluency, slightly slower)
  - Online free: Google Translate unofficial endpoint (no key)
  - Online free: Gemini free tier, `gemini-2.5-flash` or `flash-lite` (API key)
  - Online free: DeepL API Free, 500k chars/month (API key, Vietnamese supported)
  - Online free: MyMemory, ~5k chars/day anonymous (no key)
- Three region modes, user-selectable:
  1. Single fixed region (default), re-drawn with a hotkey.
  2. Multiple regions, each with its own overlay.
  3. Whole monitor auto-scan (downscaled, lower rate).
- Survive real game conditions: resolution changes, alt-tab, fullscreen
  transitions, HDR on/off, monitor sleep, static screens.

## Non-goals

- No paid APIs, no accounts, no telemetry. Online engines are free tiers only
  and off by default.
- No inpainting or style transfer: the original text is covered by a filled box,
  not reconstructed.
- No language pairs other than English -> Vietnamese in v1.
- No installer, no auto-update, no tray autostart. Run from a `.bat` in a venv.
- No audio, no voice, no subtitle timing sync.
- No process memory reading, no DLL injection, no game API hooks.

## Anti-goals (fail conditions even if the app "works")

- Visible lag that makes reading slower than reading English.
- Overlay covering UI or eating clicks.
- OCR running at full rate on an unchanged screen (wasted GPU/CPU).
- Any interaction with the game process or its input.
- Translating the same line repeatedly instead of caching it.

## Constraints and environment facts (verified)

- Machine: RTX 3080 10 GB, Windows 11 25H2 (build 26200), Python 3.13.12 and
  3.10 available; Node 24 available but not used.
- Build 26200 supports `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`, so
  the overlay can be excluded from every capture API. This is mandatory: DXGI
  duplication returns the composed desktop, so an included overlay would feed
  its own Vietnamese back into OCR forever.
- Windows 11 also allows hiding the WGC yellow border (`draw_border=False`).
- Python 3.13 is the target: `onnxruntime-gpu` 1.31 requires >= 3.11,
  `rapidocr` 3.10 and `transformers` 5.x support 3.13. Python 3.10 would
  lose GPU onnxruntime, so 3.13 + venv.
- VRAM is shared with the game. Target <= 2 GB for OCR + MT together.
- Online engines require network and send the recognized text to a third party.
  The first time an online engine is selected, the app shows a one-time warning
  and requires confirmation. Keys live in `config.json` in plain text; the UI
  says so.
- Capture library: `windows-capture` 2.0.1 (MIT) exposes both
  `DxgiDuplicationSession` (with `.recreate()`) and `WindowsCapture` (WGC).
  One dependency instead of dxcam/bettercam, and it gives both backends.
- WGC requires Windows 10 1803+; suppressing its yellow capture border
  requires Windows 11. Accept the border on Win10.
- Model weights: NLLB-600M ~2.4 GB fp32 download, EnViT5 ~1 GB.
  Both run fp32 or fp16 on 10 GB alongside a game.
- Before product code, a feasibility gate measures DXGI, WGC, OCR on GPU, both
  local engines on GPU, peak VRAM, and overlay exclusion from capture. Its
  results can change the capture or model choices, so it runs first.

## Approaches considered

1. **NLLB only, fixed region.** Fastest to build, ~400 lines. Rejected as the
   final shape: owner wants an engine choice and multiple region modes.
2. **LLM local (Qwen2.5 3B/7B) instead of NMT.** Best Vietnamese, but 3-6 GB
   VRAM competing with the game and 0.5-1.5 s per line. Owner chose free NMT
   models instead. Kept as a documented upgrade path.
3. **Chosen: NMT engines behind one interface, capture layer with DXGI primary
   and WGC fallback, PyQt6 overlay.** Costs ~700 lines, meets latency target,
   keeps VRAM under 2 GB, and makes the engine replaceable in one file.

## Architecture

```
capture thread (10-15 Hz)
  windows-capture DXGI -> numpy BGRA frame of region(s)
  watchdog: AccessLost / black frame / stale hash -> recreate -> WGC fallback
        |
        v  (temporal tracker: per-block pixel diff, not a whole-region hash)
worker thread
  rapidocr (ONNX Runtime CUDA) -> boxes + text + confidence
  line grouping -> paragraph blocks -> normalized block text
        |
        v  (translation cache keyed by normalized block text)
local MT worker (1 task at a time)  |  HTTP pool (online, max 2)
  engine adapter: NLLB | EnViT5 | Google | Gemini | DeepL | MyMemory  (batch of blocks, generation counter)
        |
        v  (only latest generation is painted; stale results dropped)
Qt main thread
  one frameless, topmost, click-through, per-pixel-transparent window
  paint opaque box over each source block + fitted Vietnamese text
  overlay is WDA_EXCLUDEFROMCAPTURE so it never enters the next frame
```

### Modules

```
src/ocrtrans/
  app.py          entry point, wiring, hotkeys, lifecycle
  config.py       config.json load/save, defaults, validation
  capture.py      Backend enum, DxgiSession + WgcSession, watchdog, health state
  regions.py      region models (single/multi/monitor), DPI + physical-px mapping
  ocr.py          RapidOCR wrapper, CUDA->CPU fallback, line grouping
  track.py        temporal tracker: region/block diff, block states, re-OCR decisions
  translate.py    Engine protocol, local engines, online providers, cache, batching
  overlay.py      Qt overlay window, box painting, text fitting, click-through,
                  OverlayState (block matching, TTLs, stale-result policy)
  hud.py          performance HUD window (p50/p95, health state)
  bench.py        quick 5-line check; corpus/perf/OCR harnesses (see Benchmark)
  ui/settings.py  engine picker, style, rates, region mode
  ui/picker.py    fullscreen drag-to-select region editor
tests/            unit tests for pure logic
```

### Decisions made (builder's call)

- RapidOCR 3.x ONNX runtime, CUDA provider; on `onnxruntime` CUDA failure or OOM,
  log once and fall back to CPU. OCR input is downscaled so the long side is
  <= 1280 px; boxes are scaled back to physical pixels.
- Detection + recognition thresholds are config, defaults tuned on a synthetic
  dialogue screenshot in `tests/fixtures`.
- Capture self-exclusion: the overlay and HUD windows set
  `WDA_EXCLUDEFROMCAPTURE` immediately after creation; a startup self-test grabs
  one frame with the overlay painted and asserts the overlay area is absent.
  Fallback if the affinity call fails: hide overlay 1 frame before each grab.
- Lines are grouped into paragraph blocks by vertical proximity (gap <= 0.6 line
  height) and similar left edge. Each block is translated as one unit, and its
  Vietnamese is painted inside the block's bounding box, replacing all source
  lines at once. Rationale: per-line translation breaks sentences at wrap points.
  Owner chose block-level over per-line on 2026-10-23.
- Translation is cached by normalized text (trim, collapse spaces, strip control
  chars) in an LRU of 2000 entries. A cache hit paints in the same frame.
- Typewriter reveal: two identical OCR passes prove nothing, a reveal can pause at
  a word. The gate is a hold window with growth detection:
  - `safe` profile (default): text unchanged for 3 consecutive passes (~134 ms at
    15 Hz). Any growth resets the window.
  - `fast` profile: 2 consecutive passes (~67 ms), accepts premature translations,
    which are replaced in place when more text arrives.
  - Growth detection: if one text is a prefix of the other, it is treated as
    growth and resets the window even when the diff is one character.
  - OCR jitter (same words, one flipped glyph) is normalized away before
    comparison; if text keeps jittering for 5 passes the block is sent anyway.
  - Neither profile guarantees dialogue completion; that is unknowable from
    pixels. The fast profile trades a measured premature-translation rate for
    latency, and the benchmark reports that rate.
  - `pending_stable` blocks are never sent to the engine, so partial strings do
    not enter the cache.
- EnViT5 input prefix `en: `, direction fixed EN->VI; NLLB uses
  `src_lang=eng_Latn`, `forced_bos_token_id=vie_Latn`.
- Only one engine is resident in VRAM. Switching engines unloads the old model,
  frees CUDA cache, then loads the new one with a progress dialog (~5-15 s).
  The benchmark loads each engine sequentially for the same reason.
- Engine interface: `name`, `kind` (`local`/`online`), `load()`, `unload()`,
  `translate(list[str]) -> list[str]`. Online engines ignore load/unload and
  never touch VRAM. Requests use `requests` (already a transformers dependency)
  with a 3 s timeout, one retry with 200 ms backoff.
- Online providers (all free tiers, user-selectable in Settings):
  - Google unofficial: `translate.googleapis.com/translate_a/single`, no key,
    one call per block, parse the nested array response. May break or 429.
  - Gemini: `generativelanguage.googleapis.com` with a free-tier API key, all
    blocks in one JSON-array prompt, model `gemini-2.5-flash` (config allows
    `flash-lite`). Best quality of the four; slowest.
  - DeepL API Free: `api-free.deepl.com/v2/translate`, key required, sends all
    blocks in one request (`text` repeated), Vietnamese supported.
  - MyMemory: `api.mymemory.translated.net/get`, no key, one call per block,
    ~5k chars/day anonymous, lowest quality. `shortcut: no-key fallback,
    replace if daily quota hurts`.
- Online failure handling: 429 or network error marks the provider degraded for
  60 s and, if fallback is enabled (default on), switches to the last local
  engine and shows `net-fallback` in the HUD. Keys are stored in plain text in
  `config.json`; the settings UI states this.
- Unsaved game text is never persisted; only the in-memory LRU cache holds it.
- Region modes: 1 = single fixed rect; 2 = list of rects; 3 = one rect covering
  a whole monitor. Mode 3 additionally keeps only boxes with height >= 12 px and
  confidence >= 0.6, otherwise it would paint over HUD numbers. Tiling is not
  done in v1: `shortcut: whole-monitor downscale loses small text, tile if mode 3
  becomes the main mode`.
- Regions are stored as normalized fractions plus monitor id, so a resolution
  change remaps them. If the monitor aspect ratio changed by more than 2%,
  the app keeps the remap but shows "re-select region" in the HUD.
- Overlay visibility is tied to the foreground window: it is shown only when the
  window under the captured region belongs to the foreground process
  (`GetForegroundWindow` + `GetWindowThreadProcessId`), and hidden after alt-tab.
  This reads window metadata only, never game memory.
- UI language: English only. Overlay translations are Vietnamese.
- Overlay is one window spanning the union of active regions, not one window per
  box. Background default is fully opaque, settings allow 60-100%. Padding 2 px
  so no ghost of the English glyphs remains. Text color/font/outline configurable;
  default Segoe UI, white on #101014.
- Text fitting order, applied per block, never skipping a step:
  1. Wrap at the source block width, base font = 0.8 x source line height.
  2. If it does not fit, grow the box downward (subtitles: keep the bottom edge
     fixed and grow upward) up to +180% of the source block height or the screen
     margin, whichever comes first.
  3. If still tight, widen within the source text column by at most 15% per side,
     clamped to the region and screen.
  4. Only then shrink the font, down to a floor of `max(12 px, 0.7 x base)`.
  5. Ellipsis only if all of the above fail; it is logged as `fit_truncated` so
     the owner can see it in the HUD.
  Growth is refused when the new rect would overlap another block's overlay; in
  that case step 4 runs instead, so no two translations ever overlap.
  Settings expose max growth %, minimum font size and vertical alignment.
  `shortcut: font substitution (condensed face) is not used to fit Vietnamese;
  add only if truncation shows up in real games`.
- Click-through via `WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW`.
  Never steals focus, never appears in alt-tab.
- DPI: process is per-monitor-v2 aware; capture coordinates are physical pixels;
  Qt painting divides by the screen device pixel ratio.
- Hotkeys via `RegisterHotKey` only (no `SetWindowsHookEx`):
  `Ctrl+Alt+Z` enable/disable overlay, `Ctrl+Alt+R` re-select region,
  `Ctrl+Alt+M` cycle region mode, `Ctrl+Alt+H` HUD, `Ctrl+Alt+S` settings.
- Threads and cancellation (one owner per stage, no shared worker):
  1. Capture thread: grab + temporal tracker at 15 Hz. Never blocks on OCR.
  2. OCR worker: single thread, queue depth 1, newest-wins. A newer frame
     replaces a waiting task; the running one finishes (<=110 ms p95).
  3. MT worker: single thread for local engines, queue depth 1, newest-wins per
     block id. CUDA inference is not interruptible, so `max_new_tokens` is capped
     to keep one call <=250 ms.
  4. HTTP pool: `ThreadPoolExecutor(max_workers=2)` for online engines only. A 3 s
     online call can never block OCR or local MT. In-flight requests are not
     killed (blocking sockets), their results are dropped on completion; the 3 s
     timeout and the 2-worker cap bound the damage. At most 2 requests in flight.
  5. Qt main thread: paint only, no I/O, no model calls.
- Epoch cancellation: a global epoch increments on engine switch, region change,
  overlay disable, scene cut and exit. Tasks check the epoch before enqueue and
  before paint; stale results are dropped silently. Queued tasks are discarded by
  epoch, not by trying to interrupt running work.
- Engine switch order: bump epoch, stop accepting, drain both queues, wait for the
  local MT worker to go idle, unload, load, resume. The overlay clears on switch.
- Exit: bump epoch, join with a 2 s deadline, then exit; sockets die with the
  process.
- Backpressure is visible: if a task waits more than 1 s in a queue, the HUD shows
  `ocr-saturated` or `mt-saturated` so a miss is not silent.
- Frame gate: not a whole-region hash. The tracker compares the downscaled frame
  to the previous one (`absdiff` > 12 luminance on a 5x5 blur to ignore noise):
  1. Activity below 1% changed pixels -> skip OCR. A blind cap of 1.5 s forces a
     full detection regardless, and an edge-density check (Sobel) on the changed
     mask forces it earlier when text-like structures appear.
  2. Activity at or above 1% -> the region is moving (rain, particles, camera).
     Block-level check: if a known block's rect (expanded 2 px) changed more than
     2%, only that block is re-read, recognizer-only on the crop.
  3. Activity at or above 1% with no known block changed -> full detection, but
     throttled to at most every 4th pass (~267 ms at 15 Hz). This is what keeps an
     animated background from forcing OCR every frame while still finding new
     text quickly.
  4. Drift: a pass with more than 5% of pixels changed runs full detection
     immediately, so boxes follow moved text and layout changes.
  5. Scene cut (>70%) resets all states.
  6. When no block is currently `stable` (nothing on screen), full detection runs
     every pass, so new dialogue never waits on any throttle.
- Periodic full detection every 1.5 s even when idle, to catch slow drift and
  alpha fades below the diff threshold.
- The tracker never claims certainty. Worst case for newly appearing text is
  ~267 ms under heavy animation, and 1.5 s only when the screen is otherwise
  static. This is stated in the design, not hidden.
- Block states: `new` -> `pending_stable` (same text twice) -> `stable` (overlay
  shown) -> `stale` (text changed) -> `gone` (missing 2 passes). The tracker owns
  these states; the overlay only renders `stable` blocks.
- Scene cut (>70% changed) resets all states to `new` in one pass.
- `None` from capture means "no new frame", never an error.
- Health state surfaced in the HUD and settings: `ok`, `recreating`,
  `wgc-fallback`, `black-frame (HDR or exclusive fullscreen?)`, `blocked (DRM?)`.
- Black-frame detection: mean luminance < 2 for 3 consecutive new frames triggers
  the warning; app keeps trying, does not attempt any workaround.
- Config file: `%LOCALAPPDATA%\OCRTrans\config.json`; models cached under
  `%LOCALAPPDATA%\OCRTrans\models` via `HF_HOME`.
- Benchmark: the Settings check runs 5 lines as a quick smoke test and says so;
  engine ranking and latency acceptance use `python -m ocrtrans.bench`
  (quality corpus + replay harness, see below). Results are timestamped and shown
  next to each engine.
- Logging: rotating file `%LOCALAPPDATA%\OCRTrans\app.log`, INFO by default.

### Overlay lifecycle and stale translation policy

Blocks are matched between OCR passes by geometry overlap (IoU >= 0.5), so a
block keeps its identity while the text is unchanged. Every paint is tagged with
`(block_id, text_hash, generation)`. A result whose generation is older than the
current one is discarded before it can be painted.

The pipeline does not send bare lists, because an empty list is ambiguous. It
emits typed `PipelineEvent`s, each carrying `epoch` and `at`:

| kind | payload | meaning |
|------|---------|---------|
| `OCR_RESULT` | `blocks: list[BlockView]`, `observed_block_ids: list[int]` | OCR ran this pass. `blocks` are what was found; `observed_block_ids` is the scope actually scanned (all known ids after a full detect, only the crop's id after a recognizer-only pass, only ids inside a region for multi-region). The **only** event that updates presence and `miss_count`, and only for ids in `observed_block_ids`. |
| `TRANSLATION_RESULT` | `items: list[RenderItem]` | one or more blocks finished translating. Never touches `miss_count`. |
| `SKIP` | none | tracker skipped OCR; no state change, overlays stay. |
| `CAPTURE_HEALTH` | `health: Health`, `reason: str` | capture degraded (`recreating`, black, blocked). Freeze then hide. |
| `RESET` | `reason: str` | scene cut, region change, engine switch, overlay disabled. Clear immediately. |

Separating OCR from translation is required: a late translation must not look
like a fresh OCR pass, or a block that is still on screen would accumulate
misses and be removed. The scope list is required for the same reason in the
other direction: a recognizer-only crop must not make every other block look
missing. `epoch` is checked on both sides; an event from an older epoch is
ignored.

| Event | Action |
|-------|--------|
| `OCR_RESULT`, block text unchanged | Keep the existing overlay, repaint nothing |
| `OCR_RESULT`, block text changed | Mark the block `pending`; old overlay stays until the translation arrives, hidden after 1000 ms |
| `OCR_RESULT`, block absent from 2 consecutive passes | Remove that block's overlay (~134 ms at 15 Hz) |
| `OCR_RESULT` with zero blocks, 2 passes in a row | Clear all overlays (dialogue closed, menu left) |
| `TRANSLATION_RESULT` | Replace the block's overlay with the translated text; no presence bookkeeping |
| `SKIP` | Keep every overlay; no state change |
| `CAPTURE_HEALTH` (`recreating`, black frame, access lost) | Freeze current overlays for 1000 ms, then hide everything; HUD shows the health state |
| `RESET` (scene cut >70%, region change, engine switch, disabled) | Clear all overlays immediately |
| Translation error / provider degraded | No overlay for that block; if the text is new, nothing is shown rather than a wrong stale line |
| App exit | Nothing is persisted; cache dies with the process |

The 1000 ms stale cap is deliberate: showing the previous Vietnamese line for
too long is worse than showing nothing, because the player would read the wrong
dialogue. `shortcut: fixed TTLs, tune per game if dialogue pacing demands it`.

### Benchmark and latency validation

The 5-line check in Settings is a smoke test only, labeled as such. Engine choice
and the latency targets are validated by two separate harnesses, both run from
`python -m ocrtrans.bench`.

**Quality corpus** (`bench/corpus/quality.jsonl`, ~80 English lines, original
text written for this project):
- categories: dialogue short/medium/long, multi-line block, item name,
  item tooltip, menu label (uppercase), quest description, combat bark (slang),
  system message with numbers/units, proper nouns and character names.
- each line has `id`, `category`, `text`, `ref_vi` (one reference translation).
- metrics per engine: chrF++ and COMET (`wmt22-comet-da`) against `ref_vi`,
  number/unit preservation by regex, proper-noun preservation rate, output
  length ratio, empty/duplicated output count.
- plus a 10-line side-by-side view the owner rates 1-5; the rating is stored next
  to the automatic scores. Automatic scores rank, the owner decides.
- COMET is 2 GB and optional (`--no-comet` for a fast run); chrF++ always runs.

**Performance harness** (`bench/perf`):
- replays a fixed set of 30 screenshots (dialogue, tooltip, menu) through the real
  capture -> OCR -> translate -> paint pipeline at a realistic cadence
  (typewriter reveal 20 chars/s, hold 2 s, gap 0.5 s), cache disabled.
- includes a 10 s animated-background clip (rain/particles over static dialogue)
  and asserts full detections stay at the throttle floor (~4/s), not per frame.
- includes a slow-fade clip (text appears over 1.2 s under the per-frame diff
  threshold) and reports detection latency, which must stay within the 1.5 s cap.
- reports the premature-translation rate for the `fast` gate: share of blocks
  whose translation changed within 1 s of being painted.
- one warmup pass discarded, 3 measured passes.
- reports p50/p95/p99 per stage and end-to-end, updates/s, peak VRAM, and the
  first-frame time; this is what verifies the acceptance table, not the microbench.
- per-engine translation throughput is measured on 40 corpus lines unbatched
  (worst case, one block at a time), 3 reps, warmup discarded.

**OCR check** (`bench/ocr`):
- 20 rendered samples: 3 fonts, bold, outline, low contrast, 720p-scaled, small
  HUD text. Reports character error rate and box IoU against known text.
- this is how OCR settings (downscale width, thresholds) are tuned.

Results are written to `bench/report.md` and shown in Settings with timestamps so
a stale benchmark is visible.

### Capture self-exclusion protocol

The primary mechanism is `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)`
on the overlay and HUD, verified at startup and in the feasibility gate. It needs
no synchronization: excluded pixels never reach the capture API.

If the affinity call fails (older build, driver refusal), the fallback is a gate,
not ad-hoc sleeping:
- The capture thread calls `gate.acquire(deadline_ms=8)` before grabbing.
- The Qt side must hide the overlay and stop painting, then acknowledge. The call
  is a blocking queued invocation, so the capture thread waits at most 8 ms.
- If no acknowledgement arrives, that frame is **skipped** (nothing is captured),
  `exclusion_missed` increments, and the loop continues. No sleep, no busy retry.
- `gate.release()` after the grab restores the overlay and repaints the latest
  state; the overlay is never repainted between hide and grab.
- Three consecutive timeouts emit `CAPTURE_HEALTH` with `BLOCKED` and reason
  `overlay-exclusion-timeout`, and the pipeline pauses with a visible error.
- The fallback costs at most one skipped frame and cannot capture the overlay;
  the worst case is lower update rate, not a feedback loop.

### Known limits, to be stated in the UI

- True exclusive fullscreen and DRM-protected frames cannot be captured by any
  Windows API. Recommendation shown: use borderless windowed, disable HDR.
- WGC yellow border on Windows 10 cannot be hidden.
- Overlay is a desktop window. Games that block overlays will not show it.
- No guarantee about anti-cheat policy. No injection, no memory access, no input
  hooks are used, which keeps the app in the same class as Discord/OBS overlays.
- Online engines send recognized game text to a third-party server. Google's
  endpoint is unofficial and may stop working without notice. Free quotas can
  run out mid-session; the app then falls back to the last local engine.
- API keys are stored unencrypted in `config.json`.

## Performance budget (acceptance)

**What is measured.** One sample starts at the pass where the tracker first sees
the final text of a block (before the stability gate) and ends when that block's
Vietnamese is painted. Samples are kept in two rings of 200 (one for cached, one
for uncached); the HUD shows p50 and p95 live, the log records p50/p95/p99 per
session. Cached and uncached samples are counted separately and never mixed.

**Sampling assumptions.** Capture runs at 15 Hz by default (config 5-30 Hz), so
one interval is 67 ms. The typewriter stability gate waits one further interval:
a block is translated when two consecutive passes return the same text, which
costs about 67 ms after the text stops changing. The game's own reveal animation
is not counted.

| Stage | Budget (p50) | Budget (p95) |
|-------|--------------|--------------|
| capture + temporal diff | <= 5 ms | <= 12 ms |
| stability gate, safe / fast | 134 / 67 ms | 140 / 70 ms |
| OCR, dialogue box <= 1280 px | <= 60 ms | <= 110 ms |
| translation, cached | < 1 ms | < 2 ms |
| translation, NLLB cold | <= 150 ms | <= 250 ms |
| translation, EnViT5 cold | <= 250 ms | <= 400 ms |
| translation, Google/DeepL/MyMemory | <= 400 ms | <= 900 ms |
| translation, Gemini | <= 1200 ms | <= 2200 ms |
| paint + font fit | <= 5 ms | <= 12 ms |
| **end-to-end, safe, NLLB** | **<= 450 ms** | **<= 550 ms** |
| **end-to-end, safe, EnViT5** | **<= 550 ms** | **<= 700 ms (P99 <= 900)** |
| **end-to-end, fast, NLLB** | **<= 300 ms** | **<= 450 ms** |
| **end-to-end, local cached** | **<= 200 ms** | **<= 220 ms** |
| **end-to-end, Google/DeepL/MyMemory** | **<= 700 ms** | **<= 1200 ms (P99 <= 1800)** |
| **end-to-end, Gemini** | **<= 1500 ms** | **<= 2600 ms (P99 <= 4000)** |
| idle CPU while screen static | <= 2% of one core | - |
| VRAM total (OCR + local MT) | <= 2 GB | - |

Every end-to-end row is the arithmetic sum of its stage p95 budgets plus the gate,
so no row can be invalidated by one stage alone: safe NLLB = 140+12+110+250+12 ≈
524, safe EnViT5 ≈ 674, Google ≈ 1168, Gemini ≈ 2468. The `fast` gate has no p95
target for EnViT5; if it misses 500 ms the app falls back to reporting it.

Online engines cannot be guaranteed on a bad network; the HUD shows the real
numbers and the p95, and the setting text says so. If the P95 target is missed in
the real-game check, the first knob is capture rate (15 -> 10 Hz lowers OCR load
but adds 33 ms to the stability gate), the second is OCR downscale width.

## Testing

- Unit tests (pytest): block grouping and ordering, text normalization and cache,
  box scaling under DPI, engine prefix/token config, config validation,
  watchdog state transitions with a fake backend,
  temporal tracker (animated background with static text -> zero OCR calls,
  one block changed -> recognizer-only on that crop, box drift -> full
  detection, scene cut -> reset, periodic refresh fires on schedule),
  typewriter stability gate (partial text never escapes), online adapters against
  mocked HTTP (429, timeout, malformed body), degraded-provider fallback,
  overlay lifecycle (block IoU matching, miss TTL, 1000 ms stale cap, scene-cut
  clear, outdated generation discarded, capture-unhealthy freeze then hide),
  text fitting (growth before shrink, font floor, collision refusal, screen clamp,
  ellipsis only as last resort, Vietnamese diacritic measurement),
  latency stats (p50/p95/p99 over a ring buffer, cached and uncached samples
  never mixed), concurrency (epoch drops stale results, queue newest-wins, a slow
  online request never blocks OCR, engine switch drains and reloads).
- Self-capture test: overlay painted, grab frame, assert overlay pixels absent.
- Manual smoke test: a local window showing a game screenshot with dialogue text;
  verify overlay covers English, Vietnamese fits, no click capture, alt-tab and
  resolution change recover.
- Real game check by the owner: one dialogue-heavy game, borderless mode.
- Online check: one live call per provider, key optional; record latency and any
  quota error in the log.

## What is left to the builder

- Exact RapidOCR 3.x parameter names for CUDA provider and box output shape.
- Whether `.recreate()` on the DXGI session is enough or the session object must
  be rebuilt; verify against the installed 2.0.1 API.
- Tuning of the fitting constants (base font ratio, growth cap) on real screenshots.
- Threshold tuning for the tracker (1%, 2%, 5%, edge energy) on real game footage.
- `max_new_tokens` per engine and OCR downscale width after measuring.
