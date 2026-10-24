# Text-first overlay pipeline (rework)

Date: 2026-10-24
Status: design, awaiting approval
Supersedes: the pixel-diff scheduling of `TemporalTracker` and the miss-count
lifecycle in `OverlayState`.

## 1. Why

Two reproducible failures come from the same design mistake: OCR scheduling
depends on per-pixel change, and the overlay decides what to hide from an
incomplete list of blocks.

Evidence from the code and the user's run:

1. **Stale translation never clears.** `_run_ocr` (`pipeline.py:369-392`)
   returns `observed` built from the blocks that still exist. A block that
   disappears is already removed from `observed` by the pass that reports the
   miss, so `OverlayState._on_ocr` (`lifecycle.py:113-129`) never increments
   `miss` for it. The block stays `shown` with its old translation until a
   RESET. This is the "overlay does not clear after the text changed" bug.
2. **Hang / log flood on moving backgrounds.** `TemporalTracker.update`
   (`track.py:145-149`) treats "70% of the pixels changed" as a scene cut and
   forces a full detection every frame; a video, water, smoke or a moving
   camera reaches that easily. With text present, `track.py:163-175` also
   requests a crop for every block whose area changed, so the OCR thread never
   idles. Measured on the user's run: OCR p50 313 ms / p95 852 ms, warnings at
   ~10-14/s, `text 0` while a subtitle was on screen.
3. **Fallback capture stops recognition.** `pipeline.py:296-307` discards the
   frame and resets the tracker whenever `health is not Health.OK`. On a
   `WGC_FALLBACK` session no OCR runs at all, so the overlay freezes or hides.
4. `group_blocks` merges paragraphs; the overlay keeps one cover per paragraph,
   which is right. The cover-fit work (`fit.py`, DP wrapping) stays.

The user's requirement: it must work on moving video backgrounds, not only on a
static game UI.

## 2. Goals, non-goals, anti-goals

Goals:

- OCR a region of any content correctly while the background moves: subtitles,
  cutscenes, video, HUD.
- The overlay never shows a translation for text that is no longer on screen.
  A real text change hides the old cover at once; a block that stops being
  detected is hidden after `ttl` (~2 passes), so a single false-negative pass
  on a moving background does not strobe the overlay.
- Bound the cost: a fixed scan budget, independent of the number of blocks and
  of how much of the screen changes. No unbounded crop fan-out.
- Keep the translation path, cache, engine switching, cover painting, exclusion
  gate, and free/offline defaults.
- Add at least one free offline translation option besides NLLB/EnViT5, after
  measuring it (research included below).

Non-goals:

- Per-frame (15 fps) text latency. Scanning at 2-4 passes/s is enough for
  subtitles and game dialogs; a typewriter line appears within 1-2 passes.
- Detecting tiny UI text on a whole-screen region. The user selects the text
  area; tiling is a bounded fallback, not the default.
- Cloud translation by default. Online engines stay behind the consent checkbox.
- Rewriting the overlay painting, font fitting or the translation cache.

Anti-goals (a result that is still a failure even if it runs):

- Showing the previous translation over new source text, ever.
- A block that stays painted after its text is gone.
- OCR work that grows with the number of tracked blocks or with background
  motion.
- A capture fallback that silently stops recognition.

## 3. Approaches considered

**A. Patch the current skeleton.** Keep pixel diff + crops; replace miss counts
with a per-block TTL, add an OCR budget, require 2-3 consecutive scene-cut
frames. Least code, keeps the existing tests. But motion still drives crop
fan-out, and the scene-cut threshold remains a guess tuned per content.

**B. Text-first (chosen).** Remove pixel diff and crop tracking from the
scheduling path. Scan the whole region at a fixed cadence; compare the detected
**text set** between passes; keep every block with a wall-clock `last_seen` and
hide it when it is absent. Background motion is irrelevant by construction.

**C. Hybrid.** B plus targeted crops for unstable blocks. Best accuracy on fast
changing text, but brings back the "when to crop" problem and the flood risk.
Can be added on top of B later without changing the model.

B is chosen because it removes both root causes instead of tuning around them,
and because its cost is measurable and testable (passes per second, OCR ms).

## 4. Architecture

### 4.1 Components

- `pipeline.Pipeline` — orchestration; keeps capture, OCR and MT threads.
- `blocks.BlockMemory` (new, replaces `TemporalTracker` scheduling) — block
  identity and text stability across scan passes. No pixels.
- `scan.ScanBudget` (new, small) — decides when the next pass may run, from
  measured OCR duration and the configured fps.
- `lifecycle.OverlayState` — rewritten as a time-based state machine fed by
  complete scan snapshots.
- `ocr.RapidOcrEngine.read` — unchanged signature for a full pass; the crop
  variant is removed.
- `translate` — unchanged plus the new offline engine from section 7.

### 4.2 Scan loop (capture thread)

```
loop:
  wait until budget allows a pass            # scan_interval
  frame, health = capture.grab(region)
  if frame is None: emit health, sleep, continue
  frame = maybe_downscale_for_wake_check(frame)   # 320 px gray, wake only
  wake = changed_pixels_ratio(frame) > 0.01       # optional, see 4.5
  if health != OK: emit CAPTURE_HEALTH            # do NOT reset, do NOT skip
  emit SCAN_START(frame, health) -> ocr queue (depth 1, newest wins)
```

- Capture runs at `fps` (settings, default 15) only to keep a fresh frame and
  drive the cheap wake check; a pass is queued when the budget permits.
- `health != OK` no longer skips the frame. The frame is used, the health is
  reported to the overlay (which holds the last cover for 1 s, then hides).
- No `tracker.reset()` on fallback. A reset happens only on region change,
  enable/disable, engine change, or stop.

### 4.3 OCR pass (ocr thread)

One pass = one full-region detection + recognition, plus grouping:

1. `boxes = ocr.read(frame)` (det + rec, existing tuned parameters).
2. `groups = group_blocks(boxes)`; one Box per paragraph, `line_px` from the
   median line height.
3. `blocks = memory.observe(groups, now)` — identity + stability (4.4).
4. Emit `OCR_RESULT` with a **complete snapshot**: the blocks matched in this
   pass plus any block still inside its `ttl` (a missed pass keeps it for one
   more pass). `blocks=[...]`, `complete=True`. There is no
   `observed_block_ids` list to misinterpret; a block missing from the snapshot
   has expired, and the overlay hides it.
5. Submit stable, changed texts to MT (unchanged selection: stable, >= 3 chars,
   not already submitted for this epoch, max 12 per pass ordered by area).

Pass duration is recorded; `ScanBudget` uses it to stretch the interval.

Tiling rule (bounded): if the region's long side is > 1600 px, the scan runs on
one capped frame (det long side 1280, as today). If a pass returns **zero**
blocks and the region is that large, the next pass may run tiled (max 2x2 tiles
of 1280 px, 64 px overlap, boxes mapped back, duplicates merged by IoU >= 0.5
keeping the higher confidence). At most one tiled pass every 2 s, and never
two tiled passes in a row. Rationale: dense text is read by the fast pass; the
tiled pass only rescues sparse small text, and its cost is capped.

### 4.4 BlockMemory

Same `TrackBlock` shape minus pixel fields; matching keeps `_match_score`
(IoU with containment floor). The stability window becomes **time-based**
(`stable_after_s`: fast 150 ms, safe 300 ms), because the pass rate is no longer
15/s; "3 passes" would mean ~1 s at the new cadence. `stability_profile` keeps
its name in the config and control window, but maps to seconds.

```python
@dataclass
class TrackBlock:
    id: int
    rect: Rect
    text: str                # last text that was accepted
    state: BlockState        # PENDING | STABLE
    first_seen: float        # start of the current text
    last_seen: float         # monotonic, set on every match
    line_px: int

class BlockMemory:
    def observe(self, boxes: list[Box], now: float) -> list[TrackBlock]:
        # match each box to the best block by rect score, else new id
        # same text (norm equal) or similarity >= 0.9: keep text, last_seen = now
        # different text: replace text, first_seen = now, state = PENDING
        # PENDING -> STABLE after stable_after_s of the same text
        # return matched blocks + blocks still inside ttl
```

Text similarity: after normalisation, `difflib.SequenceMatcher.ratio() >= 0.9`
is the same text (a one-character OCR wobble). Only a different text resets
stability and re-submits MT.

Snapshot contract: the returned list is matched blocks **plus** blocks whose
`last_seen` is within `ttl = max(2 * scan_interval, 400 ms)`. A block absent
from that list is expired, and the overlay removes it. One false-negative pass
at 3 passes/s therefore keeps the cover for ~0.6 s instead of strobing it.

### 4.5 ScanBudget

```python
interval = max(1.0 / fps, 0.25, 2.0 * last_ocr_ms / 1000)
if consecutive_empty >= 3: interval = max(interval, 1.0)
if wake and consecutive_empty:                 # cheap 320px gray diff
    interval = min(interval, 0.25)             # text may have appeared
```

- 171 ms OCR on a synthetic 2560 scene -> ~3.3 passes/s.
- An empty screen settles at 1 pass/s (wake check still runs at fps).
- `wake` uses absolute pixel change only as a **wake signal**; it never selects
  crops, never resets blocks, never forces a full detect.
- The wake check is optional in v1: if it adds complexity, start with the fixed
  interval of 1 s on empty output (decision left to the builder, default: keep
  it, it is 10 lines on a 320 px gray frame).

### 4.6 Overlay state machine

`OverlayState` keeps `{block_id: _Block(rect, source, translated, line_px,
last_seen)}` and is fed by complete snapshots.

| event | action |
|---|---|
| `RESET` | clear everything, return `[]` |
| `OCR_RESULT` (complete) | for each view: create/update. If the normalised source is the same or >= 0.9 similar, keep `translated` (a wobble must not clear the cover) and update rect/line_px/last_seen. If it is a different text, set `translated=""` and `suppressed=False` immediately. For every stored id **not** in the snapshot: remove it. Return render items. |
| `TRANSLATION_RESULT` | apply only if `normalise(item.source) == normalise(stored.source)`; otherwise drop |
| `CAPTURE_HEALTH` | `WGC_FALLBACK` is a working frame: ignore it. `RECREATING`, `BLACK`, `BLOCKED`: keep painting for up to 1 s, then set `suppressed=True`. Itsend of hold (health OK or the next OCR snapshot) clears `suppressed`. |
| no OCR event for >= 1.5 s | `suppressed=True` (watchdog on the drain timer) |
| `SKIP` | nothing (the event is removed in the new pipeline) |

`suppressed` is a render gate, not a data wipe: `translated` and `source` stay in
memory, so the same text repaints as soon as a snapshot or healthy frame
returns, and the MT dedupe (`pipeline._last_mt`) does not need to be reset.
Only snapshot absence removes a block for good.

Rendering: a block paints only when `translated` is non-empty and not
suppressed; cover = block rect, text fitted inside. A block whose source
changed paints nothing until the new translation arrives. That is the explicit
trade chosen: a short gap instead of a wrong overlay.

### 4.7 Status and instrumentation

`stats()` (control window + HUD + bench):

- keep: `ocr_passes`, `ocr_errors`, `mt_errors`, `blocks`, `p50`, `p95`,
  `ocr_p50`, `ocr_p95`, `inflight`.
- add: `scan_rate` (passes in the last 5 s), `rendered` (items painted),
  `hidden_stale` (blocks removed because they left the snapshot), `tiled`
  (passes that used tiling), `last_scan_ms`.
- `ocr_queue` becomes a boolean "pending frame" (depth 1) — keep the key.

Status line: `OCR 190 pass (3.1/s) · text 2 · shown 2 · err 0/0 · p50 171 ms`.

## 5. Failure handling

- OCR exception: counted, rate-limited log, keep the previous overlay, continue
  (already implemented).
- Translation error: service counter + fallback engine (already implemented).
- Capture `None`: report health, keep the last overlay up to 1 s.
- Capture fallback: recognition continues; health is only informational.
- No frames for 1.5 s: the overlay watchdog clears the screen.
- Region change / engine change / pause: RESET (unchanged).

## 6. Tests

Unit:

- `test_overlay_hides_a_block_that_left_the_snapshot` — the bug in 1.1, written
  to fail before the rewrite. Runs past the TTL (fake clock), then asserts the
  removed block produced no render item.
- `test_overlay_keeps_the_cover_through_one_missed_pass` — the strobe guard.
- `test_overlay_clears_translation_when_source_changes` and
  `test_overlay_keeps_translation_for_a_near_identical_source` (>= 0.9).
- `test_overlay_drops_translation_for_old_source`.
- `test_overlay_fallback_capture_keeps_painting` (WGC_FALLBACK is not a hold)
  and `test_overlay_black_capture_hides_after_one_second`.
- `test_overlay_watchdog_suppresses_without_wiping` — the same source returns
  and paints without a new translation request.
- `test_block_memory_keeps_identity_across_passes` and drops after TTL.
- `test_block_memory_stability_uses_time_not_passes` (150/300 ms profiles).
- `test_scan_budget_never_exceeds_budget` (fake clock, measured 300 ms pass ->
  interval >= 600 ms) and empty-output backoff.
- Pipeline: a moving background must not add passes; a static frame with text
  must keep the same block id and text; a changed text must produce a snapshot
  without the old text.

Integration (no game, real OCR):

- `test_video_background_subtitle`: synthetic moving background (random noise /
  scrolling gradient) with a fixed subtitle; OCR finds it and the overlay keeps
  it across passes. Then change the subtitle; assert the old translation is
  gone from the rendered items in the first snapshot after the change.
- `test_empty_screen_does_not_spin_ocr` updated to the new cadence (<= 4
  passes / 1.5 s).

Real suite (`--run-real`, needs the game): window capture, subtitle-like region,
region with motion; numbers for scan rate, OCR p50/p95, stale count.

## 7. Free translation engines (research)

Current: NLLB-600M (default, offline, ~1.15 GB VRAM) and EnViT5 (offline,
~0.59 GB); online Google (unofficial endpoint), MyMemory (no key), DeepL
(key), Gemini (key), all behind the consent checkbox.

Measured/found:

| Option | Free | Offline | Size / cost | Quality signal | Verdict |
|---|---|---|---|---|---|
| NLLB-600M | yes | yes | 1.15 GB VRAM | best in current bench | keep default |
| EnViT5 | yes | yes | 0.59 GB VRAM | close second | keep |
| Argos en->vi (`translate-en_vi`, 1.9) | yes | yes | ~100 MB, CPU | below NLLB, above literal MT | add as light engine |
| LibreTranslate (self-host) | yes | yes | Docker + Argos models | = Argos | document, optional endpoint |
| Helsinki opus-mt-en-vi | yes, Apache-2.0 | yes | ~75 MB | BLEU 37.2 on Tatoeba (chrF 0.542) | consider, likely below Argos/NLLB |
| Google (unofficial) | yes | no | rate limited, ToS grey | strong | keep behind consent |
| MyMemory | yes | no | 5k chars/day, no key | weak | keep as fallback |
| DeepL Free | 500k chars/month | no | key | strongest | already implemented |
| Azure Translator F0 | 2M chars/month | no | account | strong | not now (account setup) |
| Gemini free tier | quota | no | key | strong, slow | already implemented |
| Groq free tier (Llama) | quota | no | key | variable, fast | not now |
| LibreTranslate public mirrors | yes | no | rate limited, unstable | = Argos | not now |

Plan for this rework: add **Argos** as an optional offline engine
(`argostranslate`, model downloaded on first use) and expose
**LibreTranslate** as an optional endpoint string (`http://localhost:5000`),
reusing the existing online-engine interface. Keep NLLB as the default. No new
required dependency: both are extra, and the app must run without them.

Decision left to the builder: whether Argos goes in now or in a follow-up.
Recommendation: now, because it is the only free offline engine that fits a
low-VRAM machine, and it reuses `_OnlineBase`-like HTTP or a thin local class.

## 8. Migration and removals

- Delete `track.TemporalTracker.update` pixel logic and `Decision.crops`;
  keep `_match_score`/`_apply_text` in the new `blocks.py`.
- Delete `ocr.RapidOcrEngine._read_crops` and the `rects` argument (no callers
  after the pipeline rewrite); update `tests/test_ocr.py`.
- Rewrite `lifecycle.OverlayState`; `RenderItem`/`PipelineEvent` lose
  `observed_block_ids`, gain `complete`.
- `Pipeline.stats` gains fields above; `ui/control.set_status` updated.
- Keep `fit.py`, cover painting, exclusion gate, fonts, engine switching, cache.
- Update README (behaviour, metrics, translation engines) and the design docs.

## 9. Risks

- Scan latency to *new* text: up to ~1 s on an empty screen with backoff. The
  English source is visible during that gap. Mitigation: wake signal.
- Sparse tiny text on a full-screen region is still weaker than a tight region;
  tiling is bounded, not free.
- Selecting a huge region raises OCR cost: 171 ms at 1280 capped, ~2x when
  tiled. The region picker guidance matters.
- A block that keeps the same rect but changes text between passes may be
  matched as "same block" with a fuzzy text; the similarity >= 0.9 rule already
  treats near-identical text as stable, and the exact-text equality check before
  painting prevents showing a translation for different text.

## 10. Decisions I made (check these while reading)

1. Hide, never show stale: a real source change hides the block immediately,
   even if this makes a 200-800 ms gap until the new translation. A <= 10%
   character difference counts as the same text and keeps the cover (OCR
   wobble must not strobe).
2. Blocks leave the snapshot only after `ttl` (`max(2 * interval, 400 ms)`), so
   one false-negative pass at ~3 passes/s keeps the cover ~0.6 s.
3. `WGC_FALLBACK` is a working frame and does not hide anything; only
   `RECREATING`/`BLACK`/`BLOCKED` engage the 1 s hold, and that hold suppresses
   painting without wiping the translation.
4. Empty-screen backoff is driven by scan results, with a cheap pixel wake
   signal (not a scheduler).
5. Tiling only as a bounded rescue for large regions, never the default.
6. `stability_profile` stays user-facing but becomes a time window (fast 150 ms,
   safe 300 ms), not a pass count, so it survives a change of scan cadence.
7. Argos is added as an optional engine now; LibreTranslate documented as an
   endpoint, not shipped as a server. Both must be removable without breaking
   the app.
8. A snapshot that arrives while the overlay is suppressed repaints the same
   text without a new translation request; `pipeline._last_mt` is never cleared
   by suppression.
