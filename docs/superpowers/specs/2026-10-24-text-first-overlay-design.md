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
- Treating a similar-but-different source as the same text ("100 HP" vs "10 HP").
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
  if pending OCR frame: sleep 16ms; continue    # never replace a frame
  frame, health = capture.grab(region)          # at fps while idle
  if frame is None: emit health, sleep to next tick; continue
  wake = changed_pixels_ratio(320px gray) > 0.01
  interval = budget.interval(fps, last_ocr_ms, empty_count, wake)   # single owner
  if now - last_ocr_end < interval: sleep to next tick; continue
  scan_id += 1
  emit SCAN_START(frame, health, scan_id) -> ocr queue (depth 1)
```

- **At most one OCR pass in flight.** The capture thread does not grab while a
  frame is pending, so the pending frame is never replaced. This makes latency
  predictable: `latency <= interval + ocr_ms + mt_ms`.
- The wait is measured from the **end** of the previous pass
  (`now - last_ocr_end`), and it is recomputed every idle tick from the same
  `ScanBudget.interval` the tests call. A wake therefore shortens the wait to
  the budget floor, never below it.
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
4. Emit `OCR_RESULT` with a **complete snapshot** and a monotonic `scan_id`:
   the blocks matched in this pass plus any block still inside its `ttl` (a
   missed pass keeps it for one more pass). `blocks=[...]`, `scan_id=n`,
   `complete=True`. There is no `observed_block_ids` list to misinterpret; a
   block missing from the snapshot has expired, and the overlay hides it.
5. Submit stable, changed texts to MT. Each item carries the exact source text
   and the `scan_id` that produced it. Selection: STABLE, >= 3 chars, exact
   text not already translated/in flight for this epoch, max 12 per pass by area.
6. Record `last_ocr_end = clock()` and `last_ocr_ms = last_ocr_end - pass_start`;
   the budget uses both.

Tiling rule (bounded): if the region's long side is > 1600 px, the scan runs on
one capped frame (det long side 1280, as today). If a pass returns **zero**
blocks and the region is that large, the next pass may run tiled (max 2x2 tiles
of 1280 px, 64 px overlap, boxes mapped back, duplicates merged by IoU >= 0.5
keeping the higher confidence). At most one tiled pass every 2 s, and never
two tiled passes in a row. Rationale: dense text is read by the fast pass; the
tiled pass only rescues sparse small text, and its cost is capped.

### 4.4 BlockMemory

Block identity uses the rect score only (`_match_score`: IoU with a containment
floor). Text is compared **exactly** after normalisation; there is no similarity
threshold anywhere in the validity path.

Rationale: "You have 100 HP" and "You have 10 HP" are ~95% similar and mean
different things. A fuzzy rule that keeps the old translation would paint
"100 HP" over "10 HP". Fuzzy matching may only decide *which block is this*,
never *whether a translation is still valid*.

```python
@dataclass
class TrackBlock:
    id: int
    rect: Rect
    text: str                # exact normalised text of the last accepted read
    state: BlockState        # PENDING | STABLE
    first_seen: float        # when this exact text first appeared
    last_seen: float         # monotonic, set on every match
    line_px: int

class BlockMemory:
    def observe(self, boxes: list[Box], now: float) -> list[TrackBlock]:
        # match each box to the best block by rect score, else new id
        # text equal: keep it, stable after stable_after_s, last_seen = now
        # text different by any character (digits included):
        #     text = new, first_seen = now, state = PENDING
        # return matched blocks + blocks still inside ttl
```

- Stability is time-based and exact: `stable_after_s` (fast 150 ms, safe
  300 ms). A wobbling OCR read never becomes STABLE, so no MT is submitted and
  nothing is painted until the read settles. That is the cost of not guessing:
  the English source stays visible instead of showing a wrong translation.
- `stability_profile` keeps its name in the config and control window but maps
  to seconds.
- Snapshot contract: matched blocks **plus** blocks whose `last_seen` is within
  `ttl = min(max(2 * interval, 0.4 s), 1.0 s)` (the cap stops a slow pass from
  pinning a stale cover for seconds). A block absent from that list is expired
  and the overlay removes it. One false-negative pass at ~3 passes/s keeps the
  cover ~0.6 s instead of strobing it.

### 4.5 ScanBudget

One owner: `ScanBudget.interval(fps, last_ocr_ms, empty_count, wake) -> float`.
Tests call it and assert its return value; they never re-derive the formula, so
code and test cannot drift apart.

```python
MIN_INTERVAL_S = 0.25
OCR_HEADROOM = 1.5
EMPTY_BACKOFF_S = 1.0
EMPTY_PASSES = 3

base = max(1.0 / fps, MIN_INTERVAL_S, OCR_HEADROOM * last_ocr_ms / 1000)
if empty_count >= EMPTY_PASSES and not wake:
    base = max(base, EMPTY_BACKOFF_S)   # wake cancels the backoff, not the floor
interval = base
```

Pinned cases (fps 15):

| last_ocr_ms | empty_count | wake | interval |
|---|---|---|---|
| 300 | 0 | no | 0.450 s |
| 300 | 0 | yes | 0.450 s |
| 300 | 3 | no | 1.000 s |
| 300 | 3 | yes | 0.450 s |
| 852 | 3 | no | 1.278 s |
| 852 | 3 | yes | 1.278 s |
| 20 | 0 | no | 0.250 s |
| 20 | 5 | no | 1.000 s |

Every row is a parametrised test case against `ScanBudget.interval`; the table
and the pseudocode are checked against the same function, so a code change that
breaks a row fails the suite.

- Origin is the **end** of the previous pass: the loop waits until
  `now - last_ocr_end >= interval`.
- The floor is `1.5 * last_ocr_ms`: an 852 ms pass means a >= 1.28 s interval
  even when a wake fires. A wake can remove the extra 1 s empty-screen backoff,
  never race the OCR worker.
- At most one pass is in flight (4.2), so the interval never overlaps OCR and
  the capture thread never replaces a pending frame.
- 171 ms measured -> ~3.3 passes/s; 852 ms -> ~0.8 passes/s, the honest
  throughput of that machine.
- Wake is a signal only: no crops, no reset, no forced detect.

### 4.6 Overlay state machine

`OverlayState` keeps
`{block_id: _Block(rect, source, translated_for, translated, line_px)}` plus
`last_scan_id` and `suppressed_at_scan`, and is fed by snapshots ordered by
`scan_id`.

| event | action |
|---|---|
| `RESET` | clear everything, `suppressed=False`, return `[]` |
| `OCR_RESULT` | ignore when `event.scan_id <= last_scan_id` (stale completion). Otherwise, per view: create/update. If the normalised source differs from the stored source by **any** character, set `translated=""`, `translated_for=""`. Apply rect/line_px/last_seen. Remove stored ids absent from the snapshot. Then `last_scan_id = event.scan_id`, and if `suppressed` and `scan_id > suppressed_at_scan`, set `suppressed=False`. |
| `TRANSLATION_RESULT` | apply only when `normalise(item.source) == stored.source` exactly; set `translated`, `translated_for = item.source` |
| `CAPTURE_HEALTH` | `WGC_FALLBACK` is a working frame: ignore. `RECREATING`, `BLACK`, `BLOCKED`: keep painting up to 1 s, then `suppressed=True`, `suppressed_at_scan = last_scan_id`. Health OK does **not** unsuppress. |
| no OCR event >= 1.5 s | `suppressed=True`, `suppressed_at_scan = last_scan_id` (watchdog on the drain timer) |
| `SKIP` | removed from the event set |

- Suppression is a render gate, not a data wipe: `translated` stays in memory,
  so the same text repaints without a new MT request. But it is only lifted by
  a **snapshot with `scan_id > suppressed_at_scan`**, never by health alone.
  After a 2 s gap where the subtitle changed A -> B, the returning snapshot
  carries B; A cannot repaint because B's exact text cleared A's translation
  before suppression was lifted.
- Paint condition: `translated` non-empty, `translated_for == source` (exact,
  normalised) and not suppressed.
- A one-character OCR wobble clears that block's cover (English visible) until
  a stable read and its translation arrive. Accepted trade: never show 100 HP
  when the screen says 10 HP.

Ordering invariant: one OCR worker and a monotonic `scan_id` per epoch. A pass
that finishes late can never roll state back (dropped by `scan_id`), and an MT
result can only satisfy the exact text it was requested for.

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
- No frames for 1.5 s: the overlay suppresses painting (a gate, not a wipe).
- Suppression ends on the next applied snapshot (`scan_id > suppressed_at_scan`),
  never on health alone.
- Region change / engine change / pause: RESET (unchanged).

## 6. Tests

Unit:

- `test_overlay_hides_a_block_that_left_the_snapshot` — the bug in 1.1, written
  to fail before the rewrite. Runs past the TTL (fake clock), then asserts the
  removed block produced no render item.
- `test_overlay_keeps_the_cover_through_one_missed_pass` — the strobe guard.
- `test_overlay_clears_translation_when_source_changes`.
- `test_digit_change_never_keeps_the_old_translation` — "You have 100 HP" ->
  "You have 10 HP" must drop the translation in the same snapshot (the fuzzy
  rule is gone; this is the regression test for that review finding).
- `test_overlay_drops_translation_for_old_source`.
- `test_stale_snapshot_is_ignored` — applying a snapshot with
  `scan_id <= last_scan_id` changes nothing.
- `test_overlay_fallback_capture_keeps_painting` (WGC_FALLBACK is not a hold)
  and `test_overlay_black_capture_hides_after_one_second`.
- `test_overlay_watchdog_suppresses_without_wiping` — the same source returns
  and paints without a new translation request.
- `test_watchdog_does_not_repaint_before_a_fresh_snapshot` — suppress, change
  the text, health OK, then assert nothing paints until the snapshot with the
  new `scan_id` arrives (the stale-A-after-gap case).
- `test_block_memory_keeps_identity_across_passes` and drops after TTL.
- `test_block_memory_requires_exact_text_for_stability` — a one-character
  wobble never becomes STABLE and never reuses the old translation.
- `test_block_memory_stability_uses_time_not_passes` (150/300 ms profiles).
- `test_ttl_is_capped_at_one_second` — an 852 ms pass yields ttl 1.0 s, not
  1.7 s (same fake-clock helper as ScanBudget).
- `test_scan_budget_matches_the_pinned_table` — parametrised over every row of
  the 4.5 table, calling `ScanBudget.interval` (no hand-derived arithmetic in
  the test).
- `test_budget_never_below_measured_ocr_time` — an 852 ms pass plus a wake
  still yields an interval >= 1.28 s.
- `test_pending_frame_is_not_replaced` — the capture thread does not grab while
  an OCR pass is in flight.
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
- OCR wobble on a moving background can clear a cover and re-request MT (exact
  text, no fuzzy reuse). Cost is bounded by the cache and the 12-items-per-pass
  cap; the alternative (fuzzy reuse) risks showing the wrong number or word.
- A TTL of 0.4-1.0 s means a subtitle that disappears can stay covered for up
  to 1 s on a slow pass. Chosen over strobing on every false-negative pass.

## 10. Decisions I made (check these while reading)

1. **Exact text decides validity.** Any character change (digits included)
   hides the block immediately. No similarity threshold exists in the paint
   path; "100 HP" -> "10 HP" always hides. Similarity is not used for block
   identity either (rect score is).
2. Hide, never show stale: a source change hides the block at once, even at the
   cost of a 200-800 ms gap until the new translation.
3. Blocks leave the snapshot only after
   `ttl = min(max(2 * interval, 0.4 s), 1.0 s)`, so one false-negative pass at
   ~3 passes/s keeps the cover ~0.6 s and a slow machine never pins it past
   1 s.
4. `WGC_FALLBACK` is a working frame and does not hide anything; only
   `RECREATING`/`BLACK`/`BLOCKED` engage the 1 s hold, and that hold suppresses
   painting without wiping the translation.
5. Suppression is lifted only by a snapshot with `scan_id > suppressed_at_scan`,
   never by health alone: a stale cover cannot return after a capture gap.
6. One OCR pass in flight; the next pass may start only after
   `ScanBudget.interval(fps, last_ocr_ms, empty_count, wake)` has elapsed since
   the previous pass **ended**; a wake cancels the empty-screen backoff and
   never goes below `1.5 * last_ocr_ms`.
7. Empty-screen backoff is driven by scan results, with a cheap pixel wake
   signal (not a scheduler).
8. Tiling only as a bounded rescue for large regions, never the default.
9. `stability_profile` stays user-facing but becomes a time window (fast 150 ms,
   safe 300 ms), not a pass count, so it survives a change of scan cadence.
10. Argos is added as an optional engine now; LibreTranslate documented as an
    endpoint, not shipped as a server. Both must be removable without breaking
    the app.
11. A snapshot that arrives while the overlay is suppressed repaints the same
    text without a new translation request; `pipeline._last_mt` is never cleared
    by suppression.
