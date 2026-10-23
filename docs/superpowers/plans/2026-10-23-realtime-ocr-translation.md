# Realtime OCR Screen Translation Overlay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture a game window region, OCR English text, translate it to Vietnamese with a selectable local or free online engine, and paint the translation over the original text with click-through.

**Architecture:** A capture thread feeds a temporal tracker that decides which blocks need OCR; a single OCR worker and a single local-MT worker (plus a 2-thread HTTP pool for online engines) run behind newest-wins queues and an epoch-based cancel counter. A PyQt6 overlay window paints per-block boxes, excluded from capture with `WDA_EXCLUDEFROMCAPTURE`.

**Tech Stack:** Python 3.13, numpy, opencv-python, windows-capture 2.0.1, rapidocr 3.x, onnxruntime-gpu, transformers 5.x (NLLB-600M, EnViT5), requests, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-realtime-ocr-translation-design.md`

## Global Constraints

- Windows 11 only. Python 3.13 (`py -3.13`), venv at `.venv`.
- No process injection, no game memory access, no global keyboard hooks. Hotkeys use `RegisterHotKey`.
- All user-facing text and persisted artifacts are English; Vietnamese appears only as translated output.
- Capture at 15 Hz default (config 5–30). Overlay and HUD windows set `WDA_EXCLUDEFROMCAPTURE`.
- No paid APIs; online engines are free tiers and off by default.
- One local engine resident in VRAM at a time; combined OCR + MT VRAM ≤ 2 GB.
- Control flow uses epoch cancellation, never thread interruption.
- Config at `%LOCALAPPDATA%\OCRTrans\config.json`; logs at `%LOCALAPPDATA%\OCRTrans\app.log`.
- Never persist recognized or translated game text.

## Review Focus

1. A static dialogue over an animated background (rain, particles, camera sway) must not trigger OCR every frame; test in Task 4 and the clip in Task 15.
2. The overlay must never appear in its own capture, or the pipeline loops on its own Vietnamese; verify in Task 12.
3. New dialogue appearing while an old overlay is visible must not be missed for longer than the 1.5 s blind cap; test in Task 4 and Task 11.
4. A 3 s online request must not delay OCR or local translation; test in Task 9.
5. Vietnamese must not be silently clipped or shrunk below the font floor; test in Task 10.

---

### Task 0: Feasibility gate (before any product code)

**Files:**
- Create: `scripts/feasibility.py`, `docs/superpowers/plans/2026-10-23-feasibility.md`, `requirements.txt`, `.gitignore`, `pyproject.toml`

**Interfaces:**
- Produces: `scripts/feasibility.py` — a standalone script (no `ocrtrans` import) printing one row per check with a status, and exiting non-zero unless the overall status is `PASS` or `PASS_WITH_FALLBACK`.
- Status values: `PASS` (primary path works), `PASS_WITH_FALLBACK` (fallback was executed in this same run and its numbers recorded), `BLOCKED` (no working path found), `FAIL` (a required precondition is missing). Overall status = the worst row.
- Hard checks: DXGI frame, WGC frame, OCR on GPU, NLLB translate, EnViT5 translate, overlay excluded from capture.
- Soft checks (report only): VRAM peak per engine, OCR/MT median latency, WGC border behaviour.
- Implementation may start only when the overall status is `PASS`, or `PASS_WITH_FALLBACK` with the owner's written approval in the results file.

- [ ] **Step 1: Init repo, write requirements, install**

`requirements.txt`:
```
--extra-index-url https://download.pytorch.org/whl/cu124
numpy
opencv-python
windows-capture==2.0.1
rapidocr
onnxruntime-gpu
torch
transformers
sentencepiece
requests
PyQt6
pytest
```
`pyproject.toml` uses setuptools with `package-dir = {"" = "src"}` and installs `ocrtrans`.

```powershell
git init
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

- [ ] **Step 2: Write the feasibility script**

For each check, print `name | ms | detail | PASS/FAIL`:
1. `import` all of: `windows_capture`, `rapidocr`, `onnxruntime`, `torch`, `PyQt6`, `cv2`, `transformers`.
2. `onnxruntime.get_available_providers()` must contain `CUDAExecutionProvider`.
3. DXGI: `DxgiDuplicationSession().acquire_frame(timeout_ms=1000)` returns a frame with non-zero variance.
4. WGC: `WindowsCapture(monitor_index=1)` receives a frame within 2 s.
5. OCR: draw `"Open the gate, traveler."` on a dark 800×200 image and assert RapidOCR returns it with ≥0.8 similarity; report ms.
6. MT: load `facebook/nllb-200-distilled-600M`, translate `"Open the gate."`; then unload, load `VietAI/envit5-translation`, translate the same line. Report ms and `torch.cuda.max_memory_allocated()` per engine.
7. Self-capture: PyQt6 window filled `#FF00FF`, `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)`, then capture that screen area with DXGI; assert fewer than 0.1% of pixels are magenta.

- [ ] **Step 3: Run it and record results**

Run: `.\.venv\Scripts\python scripts\feasibility.py`
Write the output table into `docs/superpowers/plans/2026-10-23-feasibility.md`, one line per check, with the measured numbers.

- [ ] **Step 4: Gate decisions (record in the same file)**

For each failed primary path, the gate must either measure a working fallback in
the same run (`PASS_WITH_FALLBACK`) or stop (`BLOCKED`). Assumed fallbacks do not
count.

- DXGI fails, WGC frame works → status `PASS_WITH_FALLBACK`, record WGC latency; Task 5 makes WGC primary.
- DXGI and WGC both fail → `BLOCKED`; do not start Task 5.
- OCR CUDA fails, CPU OCR measured → `PASS_WITH_FALLBACK`, record CPU ms; Task 6 uses `use_cuda=False`.
- Either local engine exceeds 2 GB peak VRAM → record the number; if fp16 still exceeds it → `BLOCKED` for that engine, drop it from `ENGINES`.
- Affinity fails, hide-gate exercised once and no magenta pixels captured → `PASS_WITH_FALLBACK`, record skipped frames; Task 12 implements the gate.
- Affinity fails and the hide-gate still captures overlay pixels → `BLOCKED`.
- Any model download over 3 GB → record disk usage.

- [ ] **Step 5: Record the verdict and commit**

Write `Overall: PASS | PASS_WITH_FALLBACK | BLOCKED | FAIL` at the top of the
results file, plus the owner's approval line when fallbacks are used. Commit only
when the verdict allows implementation: `chore: add feasibility gate with measured results`.

---

### Task 1: Project scaffold, config, logging

**Files:**
- Create: `src/ocrtrans/__init__.py`, `src/ocrtrans/config.py`, `src/ocrtrans/logging_setup.py`, `tests/test_config.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `Config`, `ENGINES: frozenset[str]` (`nllb`, `envit5`, `google`, `gemini`, `deepl`, `mymemory`), `load_config(path: Path) -> Config`, `save_config(cfg: Config, path: Path) -> None`, `default_config_path() -> Path`, `setup_logging() -> None`.
- `Config` fields (all with defaults): `engine: str = "nllb"`, `stability_profile: str = "safe"`, `capture_fps: int = 15`, `region_mode: str = "single"`, `regions: list[dict] = []`, `overlay_opacity: float = 1.0`, `min_font_px: int = 12`, `font_family: str = "Segoe UI"`, `text_color: str = "#FFFFFF"`, `bg_color: str = "#101014"`, `max_growth: float = 1.8`, `min_confidence: float = 0.6`, `ocr_downscale: int = 1280`, `keys: dict[str, str] = {}`, `fallback_to_local: bool = True`, `online_consent: bool = False`.
- `.venv`, `requirements.txt` and `pyproject.toml` already exist from Task 0.

- [ ] **Step 1: Write failing config tests**

```python
def test_defaults_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    cfg = load_config(p)
    assert cfg.capture_fps == 15 and cfg.engine == "nllb"
    cfg.engine = "gemini"
    save_config(cfg, p)
    assert load_config(p).engine == "gemini"

def test_unknown_engine_rejected(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"engine": "bogus"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
```

- [ ] **Step 2: Run tests, expect failure**

Run: `.\.venv\Scripts\pytest tests/test_config.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `config.py` and `logging_setup.py`**

`load_config` merges JSON over defaults, validates `engine in ENGINES`, `stability_profile in {"safe","fast"}`, `5 <= capture_fps <= 30`, `0.6 <= overlay_opacity <= 1.0`; invalid values raise `ValueError` with the field name. `setup_logging` writes a rotating file (1 MB × 2) and INFO to stderr.

- [ ] **Step 4: Run tests, expect pass, commit**

```powershell
.\.venv\Scripts\pytest tests/test_config.py -v
git add -A; git commit -m "chore: scaffold project, config and logging"
```

---

### Task 2: Latency statistics

**Files:**
- Create: `src/ocrtrans/stats.py`, `tests/test_stats.py`

**Interfaces:**
- Produces: `LatencyStats(capacity: int = 200)` where `capacity` is **per cache state** (two independent rings of at most `capacity` samples), with `add(ms: float, cached: bool) -> None`, `percentile(q: float, cached: bool | None = None) -> float`, `snapshot() -> dict` (keys `p50/p95/p99`, `n` = total samples across both rings, `cached_n`, `uncached_n`), `reset() -> None`.

- [ ] **Step 1: Write failing tests**

```python
def test_percentiles_are_separated_by_cache():
    s = LatencyStats(capacity=5)          # capacity is per ring, so both fit
    for v in [10, 20, 30, 40, 50]:
        s.add(v, cached=False)
    s.add(1, cached=True); s.add(3, cached=True)
    assert s.percentile(50, cached=False) == 30
    assert s.percentile(50, cached=True) == 2
    assert s.snapshot()["n"] == 7
    assert s.snapshot()["cached_n"] == 2

def test_ring_evicts_oldest_per_ring():
    s = LatencyStats(capacity=3)
    for v in [1, 2, 3, 4]:
        s.add(v, cached=False)
    s.add(9, cached=True)
    assert s.snapshot()["uncached_n"] == 3
    assert s.snapshot()["cached_n"] == 1
```

- [ ] **Step 2: Run to see failure** → `pytest tests/test_stats.py -v`.

- [ ] **Step 3: Implement** linear-interpolated percentile over a `collections.deque(maxlen=capacity)` per cache flag.

- [ ] **Step 4: Pass and commit** `feat: add latency statistics`.

---

### Task 3: Geometry and region modes

**Files:**
- Create: `src/ocrtrans/geometry.py`, `src/ocrtrans/regions.py`, `tests/test_geometry.py`, `tests/test_regions.py`

**Interfaces:**
- Produces: `Rect(x, y, w, h)` with `.right`, `.bottom`, `.contains_point`, `intersects(other) -> bool`; `Box(rect: Rect, text: str, confidence: float)`; `iou(a, b) -> float`, `union_all(rects) -> Rect`, `clamp(rect, bounds) -> Rect`, `expand(rect, px) -> Rect`.
- Produces: `NormalizedRegion(monitor_id: str, x: float, y: float, w: float, h: float)`, `to_physical(region, monitor: Rect) -> Rect`, `from_physical(rect, monitor: Rect) -> NormalizedRegion`, `remap(region, old_monitor: Rect, new_monitor: Rect) -> tuple[Rect, bool]` (bool = aspect changed > 2%).
- Produces: `RegionSet.from_config(cfg: Config, monitor: Rect) -> list[Rect]` handling `single`, `multi`, `monitor`.

- [ ] **Step 1: Write failing tests**

```python
def test_iou_identical_is_one():
    assert iou(Rect(0, 0, 10, 10), Rect(0, 0, 10, 10)) == 1.0

def test_rects_touching_at_edge_do_not_intersect():
    assert not intersects(Rect(0, 0, 10, 10), Rect(0, 10, 10, 10))

def test_region_roundtrip_survives_resolution_change():
    old, new = Rect(0, 0, 1920, 1080), Rect(0, 0, 2560, 1440)
    reg = from_physical(Rect(960, 810, 640, 200), old)
    rect, aspect_changed = remap(reg, old, new)
    assert rect == Rect(1280, 1080, 853, 267)
    assert aspect_changed is False

def test_region_mode_monitor_is_full_screen():
    assert RegionSet.from_config(cfg_with(mode="monitor"), Rect(0, 0, 1920, 1080)) == [Rect(0, 0, 1920, 1080)]
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** with integer rounding via `round()`; `remap` compares `old.w/old.h` and `new.w/new.h`, flags when relative diff > 0.02.

- [ ] **Step 4: Pass and commit** `feat: add geometry and region modes`.

---

### Task 4: Temporal tracker

**Files:**
- Create: `src/ocrtrans/track.py`, `tests/test_track.py`

**Interfaces:**
- Produces: `BlockState` enum `{NEW, PENDING, STABLE, STALE, GONE}`.
- Produces: `TrackBlock(id: int, rect: Rect, text: str, state: BlockState, stable_passes: int, miss_count: int, last_seen_pass: int)`.
- Produces: `Decision(full_detect: bool, crops: list[tuple[int, Rect]], skip: bool)`.
- Produces: `TemporalTracker(profile: str = "safe", block_diff_ratio: float = 0.02, region_diff_ratio: float = 0.01, drift_ratio: float = 0.05, scene_cut_ratio: float = 0.70, blind_cap_s: float = 1.5, refresh_s: float = 1.5, detect_throttle_passes: int = 4, diff_luma: int = 12, samples_per_s: int = 15)` with:
  - `update(frame: np.ndarray, now: float) -> Decision`
  - `observe(boxes: list[Box], now: float) -> list[TrackBlock]`
  - `blocks() -> list[TrackBlock]`
  - `reset() -> None`
- Stability windows: `safe` = 3 passes, `fast` = 2 passes.

- [ ] **Step 1: Write failing tracker tests** (synthetic numpy frames, 200×100)

```python
def test_animated_background_does_not_ocr_every_frame():
    t = TemporalTracker()
    base = np.full((100, 200), 30, np.uint8)
    for _ in range(3):                     # stable block: throttling applies
        t.observe([Box(Rect(20, 20, 160, 30), "HELLO THERE", 0.99)], 0.0)
    calls = 0
    for i in range(10):
        frame = base.copy()
        frame[70:100, :] = (i * 20) % 255          # heavy animated band, no text
        d = t.update(frame, 0.1 * i)
        calls += (d.full_detect or bool(d.crops))
    assert 1 <= calls <= 3                         # throttled, not 10

def test_new_text_under_activity_detected_within_throttle():
    t = TemporalTracker()
    for _ in range(3):
        t.observe([Box(Rect(20, 20, 160, 30), "OLD", 0.99)], 0.0)
    base = np.full((100, 200), 30, np.uint8)
    detections = 0
    for i in range(6):
        frame = base.copy(); frame[70:100, :] = (i * 30) % 255
        detections += t.update(frame, 0.067 * i).full_detect
    assert 1 <= detections <= 2                    # every 4th pass, not every pass

def test_blind_cap_fires_within_1_5s_of_static_screen():
    t = TemporalTracker(blind_cap_s=1.5)
    frame = np.full((100, 200), 30, np.uint8)
    for _ in range(3):                       # a stable block suppresses per-pass detection
        t.observe([Box(Rect(10, 10, 100, 20), "HELLO", 0.9)], 0.0)
    assert t.blocks()[0].state is BlockState.STABLE
    t.update(frame, 0.0)                     # baseline full detection, clock reset
    first_detect_at = None
    for i in range(1, 80):                   # deterministic clock, 50 ms steps
        now = 0.05 * i
        if t.update(frame, now).full_detect:
            first_detect_at = now
            break
    assert first_detect_at is not None
    assert 1.4 <= first_detect_at <= 1.5     # cap fired, not every pass

def test_scene_cut_resets_blocks():
    t = TemporalTracker()
    t.update(np.full((100, 200), 30, np.uint8), 0.0)   # baseline frame first
    t.observe([Box(Rect(10, 10, 100, 20), "A", 0.9)], 0.0)
    d = t.update(np.full((100, 200), 255, np.uint8), 0.1)
    assert d.full_detect is True and t.blocks() == []

def test_growth_resets_stability():
    t = TemporalTracker(profile="safe")
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO", 0.9)], 0.0)
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO THER", 0.9)], 0.07)
    assert t.blocks()[0].state == BlockState.PENDING
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO THERE", 0.9)], 0.14)
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO THERE", 0.9)], 0.21)
    t.observe([Box(Rect(10, 10, 100, 20), "HELLO THERE", 0.9)], 0.28)
    assert t.blocks()[0].state == BlockState.STABLE
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** downscale to 320 px wide, grayscale, 5×5 blur, `cv2.absdiff`; cumulative diff area resets on full detection; Sobel energy on the changed mask above 1% of pixels blocks a skip.

- [ ] **Step 4: Pass and commit** `feat: add temporal tracker`.

---

### Task 5: Capture manager with DXGI → WGC fallback

**Files:**
- Create: `src/ocrtrans/capture.py`, `tests/test_capture.py`, `scripts/capture_smoke.py`

**Interfaces:**
- Produces: `Health` enum `{OK, RECREATING, WGC_FALLBACK, BLACK, BLOCKED}`.
- Produces: `CaptureBackend` protocol: `grab(rect: Rect) -> np.ndarray | None`, `close() -> None`.
- Produces: `CaptureManager(main: CaptureBackend, fallback: CaptureBackend | None = None, clock=time.monotonic)` with `grab(rect) -> tuple[np.ndarray | None, Health]`, `health -> Health`, `close()`.
- Fallback trigger: 2 consecutive `RuntimeError`/access-loss recreates of `main`, or 3 consecutive frames with mean luma < 2.

- [ ] **Step 1: Write failing tests with a fake backend**

```python
class FakeBackend:
    def __init__(self, frames): self.frames = list(frames); self.calls = 0
    def grab(self, rect): self.calls += 1; return self.frames.pop(0)
    def close(self): pass

def test_none_frame_is_not_an_error():
    m = CaptureManager(FakeBackend([None, np.full((4, 4, 4), 9, np.uint8)]))
    assert m.grab(Rect(0, 0, 4, 4)) == (None, Health.OK)

def test_black_frames_switch_to_fallback():
    main = FakeBackend([np.zeros((4, 4, 4), np.uint8)] * 3)
    fb = FakeBackend([np.full((4, 4, 4), 9, np.uint8)])
    m = CaptureManager(main, fallback=fb)
    for _ in range(3): m.grab(Rect(0, 0, 4, 4))
    frame, health = m.grab(Rect(0, 0, 4, 4))
    assert health == Health.WGC_FALLBACK and fb.calls == 1
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement `DxgiBackend` and `WgcBackend`**

`DxgiBackend` wraps `windows_capture.DxgiDuplicationSession`, calls `acquire_frame(timeout_ms=33)`, `to_numpy()` then crops the rect; on `RuntimeError` calls `recreate()`. `WgcBackend` wraps `WindowsCapture(window_name=..., cursor_capture=False, draw_border=False)` and keeps the latest frame between callbacks.

- [ ] **Step 4: Pass and commit** `feat: add capture manager with WGC fallback`.

- [ ] **Step 5: Manual smoke** `python scripts/capture_smoke.py --window "Title"` writes one PNG; confirm it is not black.

---

### Task 6: OCR engine and block grouping

**Files:**
- Create: `src/ocrtrans/ocr.py`, `tests/test_ocr.py`, `tests/fixtures/dialogue.png`

**Interfaces:**
- Produces: `OcrResult(boxes: list[Box], text: str, elapsed_ms: float)`; `Box` is imported from `ocrtrans.geometry`.
- Produces: `OcrEngine.load() -> None`, `read(frame: np.ndarray, rects: list[Rect] | None = None) -> list[Box]` (rects = recognizer-only on crops), `close()`.
- Produces: `group_blocks(boxes: list[Box], max_gap_ratio: float = 0.6, min_confidence: float = 0.6) -> list[Block]` where `Block(rect: Rect, lines: list[str])` and `.text` joins lines with `" "`. Boxes below `min_confidence` are dropped before grouping.
- Produces: `RapidOcrEngine(use_cuda: bool = True, downscale_width: int = 1280)`.

- [ ] **Step 1: Write failing grouping tests**

```python
def test_lines_of_one_paragraph_group_in_reading_order():
    boxes = [Box(Rect(10, 34, 200, 20), "second line", 0.9),   # out of order on purpose
             Box(Rect(10, 10, 200, 20), "first line", 0.9),
             Box(Rect(10, 200, 200, 20), "menu", 0.9)]
    blocks = group_blocks(boxes)
    assert [b.text for b in blocks] == ["first line second line", "menu"]

def test_confidence_filter_drops_noise():
    boxes = [Box(Rect(0, 0, 50, 12), "x", 0.2)]
    assert group_blocks(boxes, min_confidence=0.6) == []
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement grouping** sort by `(y, x)`, union when vertical gap ≤ 0.6 × line height and left edges differ ≤ 2 × line height.

- [ ] **Step 4: Add `RapidOcrEngine`** mapping RapidOCR's `(boxes, text, scores)` to `Box`, scaling boxes from the downscaled image back to physical pixels; recognizer-only path calls RapidOCR with `use_det=False, use_cls=False` on crops.

- [ ] **Step 5: Add a GPU-marked integration test** on `tests/fixtures/dialogue.png` asserting the known sentence is recovered; `@pytest.mark.gpu`. Run with `-m gpu` manually.

- [ ] **Step 6: Pass non-GPU tests, commit** `feat: add OCR engine and block grouping`.

---

### Task 7: Translation interface, cache, local engines

**Files:**
- Create: `src/ocrtrans/translate.py`, `tests/test_translate.py`

**Interfaces:**
- Produces: `Engine` protocol: `name: str`, `kind: Literal["local","online"]`, `load() -> None`, `unload() -> None`, `translate(texts: list[str]) -> list[str]`.
- Produces: `TranslationCache(capacity: int = 2000)` with `get(key) -> str | None`, `put(key, value)`, `stats() -> dict[str, int]`.
- Produces: `normalize(text: str) -> str` (trim, collapse whitespace, strip control chars).
- Produces: `NllbEngine(model_dir: Path, device: str = "cuda")`, `Envit5Engine(model_dir: Path, device: str = "cuda")`, with `MAX_NEW_TOKENS = 96`.
- Produces: `TranslationService(engine: Engine, cache: TranslationCache, fallback: Engine | None = None)` with `translate(text: str) -> tuple[str, bool]` (text, cached), and `switch(engine: Engine) -> None`.

- [ ] **Step 1: Write failing tests with a fake engine**

```python
class FakeEngine:
    name, kind = "fake", "local"
    def load(self): pass
    def unload(self): pass
    def translate(self, texts): return [t.upper() for t in texts]

def test_cache_returns_hit_flag():
    svc = TranslationService(FakeEngine(), TranslationCache())
    assert svc.translate("hello") == ("HELLO", False)
    assert svc.translate("  hello  ") == ("HELLO", True)

def test_nllb_rejects_missing_model_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        NllbEngine(tmp_path / "nope").load()

def test_switch_refuses_while_translate_in_flight():
    busy = BlockingEngine()                       # started/release threading Events
    svc = TranslationService(busy, TranslationCache())
    t = threading.Thread(target=svc.translate, args=("hello",))
    t.start(); assert busy.started.wait(1.0)
    with pytest.raises(RuntimeError, match="engine busy"):
        svc.switch(FakeEngine())
    busy.release.set(); t.join()
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement service and local engines**

`NllbEngine` uses `AutoTokenizer` with `src_lang="eng_Latn"` and `forced_bos_token_id=tokenizer.convert_tokens_to_ids("vie_Latn")`; batched generation with `max_new_tokens=96`, `fp16` on CUDA. `Envit5Engine` prefixes each input with `"en: "` and uses `max_new_tokens=96`. `TranslationService.switch` refuses to unload while a `translate` call is in flight (raises `RuntimeError("engine busy")`), otherwise unloads, calls `torch.cuda.empty_cache()`, then loads the new one; the pipeline drains its worker before calling it.

- [ ] **Step 4: Add a GPU-marked translation smoke test** that translates `"Open the gate."` and asserts non-empty Vietnamese output with diacritics.

- [ ] **Step 5: Pass and commit** `feat: add translation service and local engines`.

---

### Task 8: Online engines and degraded fallback

**Files:**
- Create: `src/ocrtrans/online.py`, `tests/test_online.py`

**Interfaces:**
- Produces: `GoogleEngine(session: requests.Session, timeout: float = 3.0)`, `MyMemoryEngine(session, timeout=3.0)`, `DeepLEngine(api_key: str, session, timeout=3.0)`, `GeminiEngine(api_key: str, session, model: str = "gemini-2.5-flash", timeout=6.0)`; all implement `Engine` with `kind = "online"` and no-op `load`/`unload`.
- Produces: `ProviderError(Exception)`, `QuotaError(ProviderError)`.
- Produces: `OnlineRouter(primary: Engine, local_fallback: Engine | None, cooldown_s: float = 60.0)` with `translate(text) -> tuple[str, bool]` (text, degraded) and `healthy -> bool`.

- [ ] **Step 1: Write failing tests against a stub session**

```python
class StubSession:
    def __init__(self, status=200, payload=None): self.status, self.payload = status, payload
    def get(self, url, params=None, timeout=None): return Resp(self.status, self.payload)
    def post(self, url, headers=None, json=None, timeout=None): return Resp(self.status, self.payload)

def test_google_parses_nested_array():
    body = [[["Xin chào", "Hello", None, None]], None, "en"]
    assert GoogleEngine(StubSession(payload=body)).translate(["Hello"]) == ["Xin chào"]

def test_429_marks_provider_degraded_for_cooldown():
    router = OnlineRouter(GoogleEngine(StubSession(status=429)), FakeEngine())
    out, degraded = router.translate("Hello")
    assert degraded is True and out == "HELLO"
    assert router.healthy is False
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement providers**

Google: `GET https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=vi&dt=t&q=...`, parse `payload[0][0][0]`. MyMemory: `GET https://api.mymemory.translated.net/get?q=...&langpair=en|vi`, read `responseData.translatedText`. DeepL: `POST https://api-free.deepl.com/v2/translate` with `{"text": texts, "source_lang": "EN", "target_lang": "VI"}`, header `Authorization: DeepL-Auth-Key <key>`. Gemini: `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=...` with a prompt requiring a JSON array of translations, no commentary; parse the first text part as JSON.

- [ ] **Step 4: Implement router** one retry with 200 ms backoff on 5xx/timeout, cooldown 60 s on 429/401, fallback engine used when configured.

- [ ] **Step 5: Pass and commit** `feat: add free online translation engines`.

---

### Task 9: Pipeline concurrency and cancellation

**Files:**
- Create: `src/ocrtrans/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `Pipeline(capture: CaptureManager, ocr: OcrEngine, tracker: TemporalTracker, service: TranslationService, emit: Callable[[PipelineEvent], None], gate: ExclusionGate | None = None, fps: int = 15, clock=time.monotonic)` with `start()`, `stop(timeout=2.0)`, `set_regions(rects: list[Rect])`, `set_engine(engine: Engine)`, `set_enabled(bool)`, `stats() -> dict`, `epoch -> int`. `emit` only enqueues; the Qt overlay drains it, so `emit` is safe to call from any thread.
- Produces: `BlockView(block_id: int, rect: Rect, source_text: str, state: Literal["pending", "stable"])`.
- Produces: `RenderItem(block_id: int, text: str, rect: Rect, source_text: str, cached: bool)`.
- Produces: `EventKind` (str enum: `OCR_RESULT`, `TRANSLATION_RESULT`, `SKIP`, `CAPTURE_HEALTH`, `RESET`) and `PipelineEvent(kind, epoch: int, at: float, blocks: list[BlockView] = [], observed_block_ids: list[int] = [], items: list[RenderItem] = [], health: Health | None = None, reason: str = "")`.
- Only `OCR_RESULT` carries block presence, and only ids in `observed_block_ids` may be counted as missing. A full detect reports every known id plus new ones; a recognizer-only crop reports exactly that block id; multi-region reports the ids inside the scanned region.
- `TRANSLATION_RESULT` never increments a miss count.
- Produces: `ExclusionGate` protocol: `acquire(deadline_ms: int = 8) -> bool`, `release() -> None` (see Task 12 for the Qt implementation and `AffinityGate` no-op).
- Queues: OCR depth 1 newest-wins, MT depth 1 newest-wins per block id, HTTP pool `ThreadPoolExecutor(max_workers=2)` for online engines.
- `epoch` increments on `set_engine`, `set_regions`, `set_enabled(False)`, scene cut, `stop`.
- `stats()` keys: `ocr_passes`, `ocr_queue`, `mt_queue`, `inflight`, `epoch`, `fit_truncated`, `exclusion_missed`.

- [ ] **Step 1: Write failing concurrency tests**

Test helpers: `pipeline_with(...)` builds a `Pipeline` from a fake capture (returns one fixture frame then repeats), `FakeOcr`, a tracker stub, and a list sink; `p.events()` returns emitted events; `p.wait_event(pred, timeout=2.0)` blocks until a matching event. `SlowEngine(delay=None)` blocks on `started`/`release` threading Events so the test can prove a request is in flight.

```python
def test_slow_online_request_does_not_block_local_ocr():
    slow = SlowEngine(delay=3.0)          # online kind
    p = pipeline_with(engine=slow, ocr=FakeOcr())
    p.start()
    time.sleep(0.3)
    assert p.stats()["ocr_passes"] >= 2   # OCR kept running
    p.stop()

def test_engine_switch_drops_inflight_result():
    slow = SlowEngine(delay=None, name="slow", prefix="SLOW: ")   # blocks until released
    p = pipeline_with(engine=slow)
    p.start()
    assert slow.started.wait(2.0), "slow inference never started"
    p.set_engine(FastEngine(prefix="FAST: "))
    slow.release.set()
    assert slow.finished.wait(2.0), "old worker never finished"
    assert slow.unloaded_before_finish is False, "engine unloaded while in use"
    emitted = [i.text for e in p.events() if e.kind == EventKind.TRANSLATION_RESULT
               for i in e.items]
    assert not any(t.startswith("SLOW: ") for t in emitted)
    p.stop()

def test_crop_scan_reports_only_observed_ids():
    p = pipeline_with(ocr=FakeOcr(boxes=[Box(Rect(0, 0, 50, 12), "X", 0.9)]),
                      tracker_returns_crop=True)
    p.start()
    ev = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT and e.observed_block_ids)
    assert ev.observed_block_ids == [ev.blocks[0].block_id]
    p.stop()

def test_skip_is_reported_as_skip_not_empty_ocr():
    p = pipeline_with(tracker_returns_skip=True)
    p.start()
    assert p.wait_event(lambda e: e.kind == EventKind.SKIP) is not None
    p.stop()

def test_empty_ocr_result_carries_no_blocks():
    p = pipeline_with(ocr=FakeOcr(boxes=[]))
    p.start()
    ev = p.wait_event(lambda e: e.kind == EventKind.OCR_RESULT)
    assert ev.blocks == []
    p.stop()

def test_epoch_cancel_clears_queue():
    p = pipeline_with(engine=SlowEngine(delay=2.0))
    p.start(); p.set_enabled(False)
    assert p.stats()["mt_queue"] == 0
    p.stop()
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement pipeline**

Capture loop: before grabbing, if `gate` is set and `gate.acquire()` returns False, increment `exclusion_missed`, skip this frame and continue (no capture); after the grab call `gate.release()` in a `finally`. Then `frame, health = capture.grab(region)`; `None` → sleep remaining interval; `health != OK` → `tracker.reset()` and emit `PipelineEvent(kind=CAPTURE_HEALTH, health=health, epoch=epoch)`. Otherwise `decision = tracker.update(...)`; if `decision.skip` emit `PipelineEvent(kind=SKIP, epoch=epoch)` and continue. Else submit an OCR task tagged with the epoch; on completion it calls `tracker.observe` and emits `PipelineEvent(kind=OCR_RESULT, blocks=[BlockView(...)], observed_block_ids=[...], epoch=epoch)`, where the scope is every known id for a full detect, exactly the crop's id for recognizer-only, and the ids inside the scanned region for multi-region. MT results emit `PipelineEvent(kind=TRANSLATION_RESULT, items=[RenderItem(...)], epoch=epoch)`. A miss in presence is only ever recorded while handling `OCR_RESULT`. `emit` puts into a `queue.Queue`; the Qt overlay drains it, so no Qt call happens off-thread.

- [ ] **Step 4: Implement the HTTP pool** for `kind == "online"`; local engines stay on the MT worker. A slow online task never occupies the MT worker.

- [ ] **Step 5: Pass and commit** `feat: add pipeline concurrency and cancellation`.

---

### Task 10: Text fitting

**Files:**
- Create: `src/ocrtrans/fit.py`, `tests/test_fit.py`

**Interfaces:**
- Produces: `Measure` protocol: `measure(text: str, font_px: int) -> tuple[int, int]` and `line_height(font_px: int) -> int`.
- Produces: `FitResult(rect: Rect, font_px: int, lines: list[str], truncated: bool)`.
- Produces: `fit_text(text: str, source: Rect, screen: Rect, measure: Measure, *, base_ratio=0.8, max_growth=1.8, min_font_px=12, obstacles: Sequence[Rect] = ()) -> FitResult`.

- [ ] **Step 1: Write failing tests**

```python
class MonoMeasure:
    def measure(self, text, font_px): return (len(text) * font_px // 2, font_px + 4)
    def line_height(self, font_px): return font_px + 4

def test_grows_before_shrinking_or_truncating():
    r = fit_text("một hai ba bốn năm sáu", Rect(0, 500, 200, 40), Rect(0, 0, 1920, 1080), MonoMeasure())
    assert r.font_px == 32 and r.truncated is False and r.rect.h >= 40

def test_respects_font_floor_then_truncates():
    r = fit_text("x" * 400, Rect(0, 0, 100, 20), Rect(0, 0, 200, 100), MonoMeasure(), min_font_px=12)
    assert r.font_px == 12 and r.truncated is True

def test_never_overlaps_obstacle():
    r = fit_text("một hai ba", Rect(0, 0, 200, 20), Rect(0, 0, 1920, 1080), MonoMeasure(),
                 obstacles=[Rect(0, 24, 200, 200)])
    assert r.rect.bottom <= 24
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** the fixed order: wrap at source width → grow downward (subtitles anchored bottom grow upward) up to `max_growth`/screen margin → widen ≤15% per side → shrink to font floor → ellipsis. Overlap check via `intersects` against obstacles; on overlap skip growth and go to shrink.

- [ ] **Step 4: Pass and commit** `feat: add Vietnamese text fitting`.

---

### Task 11: Overlay lifecycle state machine

**Files:**
- Create: `src/ocrtrans/lifecycle.py`, `tests/test_lifecycle.py`

**Interfaces:**
- Produces: `OverlayState(stale_cap_s: float = 1.0, miss_passes: int = 2, unhealthy_hold_s: float = 1.0)` with `on_event(event: PipelineEvent, now: float) -> list[RenderItem]` (items to render, `[]` means paint nothing), `clear() -> None`, `state(block_id: int) -> str`.
- Consumes: `PipelineEvent`, `EventKind`, `BlockView`, `RenderItem` from Task 9. Ignores events whose `epoch` is older than the last accepted one. Only `OCR_RESULT` updates `miss_count`; `TRANSLATION_RESULT` merges text into the existing block.
- `state(block_id)` returns one of `"pending"`, `"shown"`, `"hidden"`.

- [ ] **Step 1: Write failing tests** (helpers: `item(block_id, text, source="X")` builds a `RenderItem`; `block(block_id, text)` builds a `BlockView`; `ev(kind, **kw)` builds a `PipelineEvent` with epoch=1 and, for `OCR_RESULT`, `observed_block_ids` defaulting to the ids present in `blocks` unless passed explicitly)

```python
def test_missing_block_removed_after_two_ocr_passes():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HELLO")]), 0.0)
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[]), 0.07)
    assert s.on_event(ev(EventKind.OCR_RESULT, blocks=[]), 0.14) == []

def test_crop_scan_does_not_miss_other_blocks():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "A"), block(2, "B")],
                  observed_block_ids=[1, 2]), 0.0)
    for t in (0.07, 0.14):
        s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "A2")],
                      observed_block_ids=[1]), t)
    assert s.state(2) == "shown"          # block 2 was never in scan scope

def test_translation_result_never_counts_as_missing():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HELLO")]), 0.0)
    for t in (0.1, 0.2, 0.3):
        s.on_event(ev(EventKind.TRANSLATION_RESULT,
                      items=[item(1, "XIN CHÀO", "HELLO")]), t)
        assert s.state(1) == "shown"

def test_skip_event_never_clears():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HELLO")]), 0.0)
    for t in (1.0, 5.0, 9.0):
        assert s.on_event(ev(EventKind.SKIP), t)

def test_stale_text_hidden_after_cap():
    s = OverlayState(stale_cap_s=1.0)
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "OLD")]), 0.0)
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "CŨ", "OLD")]), 0.05)
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "NEW")]), 0.1)
    assert s.on_event(ev(EventKind.SKIP), 0.9)
    assert s.on_event(ev(EventKind.SKIP), 1.2) == []

def test_capture_health_freezes_then_hides():
    s = OverlayState(unhealthy_hold_s=1.0)
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "A")]), 0.0)
    assert s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.RECREATING), 0.5)
    assert s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.RECREATING), 1.6) == []

def test_reset_clears_and_older_epoch_ignored():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "NEW")], epoch=2), 0.0)
    assert s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "OLD")], epoch=1), 0.1)
    assert s.on_event(ev(EventKind.RESET, reason="scene_cut", epoch=2), 0.2) == []
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** the table in the spec, including the 1000 ms stale cap and immediate clear on `RESET`. Presence is only updated for ids listed in `observed_block_ids`.

- [ ] **Step 4: Pass and commit** `feat: add overlay lifecycle state machine`.

---

### Task 12: Overlay window

**Files:**
- Create: `src/ocrtrans/overlay.py`, `src/ocrtrans/exclusion.py`, `tests/test_exclusion.py`, `scripts/overlay_selftest.py`

**Interfaces:**
- Produces: `OverlayWindow(screen: Rect, config: Config, events_queue: queue.Queue)`, with `show_items(items: list[RenderItem]) -> None`, `clear() -> None`, `set_regions(rects: list[Rect]) -> None`, `close()`. The 16 ms timer drains `events_queue` of `PipelineEvent`s and passes each to `OverlayState.on_event`; `CAPTURE_HEALTH` and `RESET` are handled by the same call.
- Produces: `AffinityGate` implementing `ExclusionGate` with `acquire() -> True` (affinity already excludes the window; no-op).
- Produces: `QtHideGate(bridge: HideBridge, deadline_ms: int = 8)` implementing `ExclusionGate`; `HideBridge.hide_and_ack(timeout_ms: int) -> bool` and `restore() -> None`. `acquire()` returns the bridge result and counts consecutive failures in `tripped` (True after 3).
- `QtHideGate` is selected only when the feasibility gate reported `PASS_WITH_FALLBACK` for exclusion.

- [ ] **Step 0: Write failing exclusion test**

```python
class FakeBridge:
    def __init__(self, ack): self.ack = ack; self.restores = 0
    def hide_and_ack(self, timeout_ms): return self.ack
    def restore(self): self.restores += 1

def test_timeout_skips_frame_and_trips_after_three():
    g = QtHideGate(FakeBridge(ack=False))
    assert g.acquire() is False and g.tripped is False
    assert g.acquire() is False and g.acquire() is False
    assert g.tripped is True

def test_success_restores_overlay():
    b = FakeBridge(ack=True); g = QtHideGate(b)
    assert g.acquire() is True
    g.release()
    assert b.restores == 1
```
- Produces: `QtMeasure` implementing `Measure` with `QFontMetrics`.
- Window flags: `Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool`; `WA_TransparentForMouseEvents`; `WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW`; `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)` after `show()`.

- [ ] **Step 0b: Implement `exclusion.py`, run the test, keep the commit for Step 3.**

- [ ] **Step 1: Implement painting** in `paintEvent`: a 16 ms `QTimer` drains the event queue, routes each `PipelineEvent` through `OverlayState`, and for each returned item computes `fit_text` with the other items' rects as `obstacles`, fills `bg_color` with `overlay_opacity`, draws wrapped lines with `QPainter.drawText`, logs `fit_truncated` once per block.

- [ ] **Step 2: Manual check** `python scripts/overlay_selftest.py` shows a fixed Vietnamese paragraph over a dummy rectangle; verify clicks pass through into a window underneath.

- [ ] **Step 3: Self-capture check** `scripts/overlay_selftest.py --verify-capture`: screenshot the region with the overlay shown and assert the overlay pixels are absent; exit code 0 on pass. Run it twice, once normally and once with `OCRTRANS_FORCE_HIDE_GATE=1` to exercise the fallback; in the fallback run assert the overlay is never captured and `exclusion_missed` stays bounded (< 5 over 60 frames). Then commit `feat: add click-through overlay excluded from capture`.

---

### Task 13: Settings UI, hotkeys, region picker

**Files:**
- Create: `src/ocrtrans/ui/settings.py`, `src/ocrtrans/ui/picker.py`, `src/ocrtrans/hotkeys.py`, `tests/test_hotkeys.py`

**Interfaces:**
- Produces: `HotkeyManager` with `register(map: dict[str, int], callback: Callable[[str], None])` using `RegisterHotKey` and a `WM_HOTKEY` message loop; `unregister_all()`.
- Produces: `SettingsDialog(config: Config, on_save: Callable[[Config], None])` with engine list, per-engine benchmark result row, key fields for Gemini/DeepL, `safe`/`fast` profile, capture fps, region mode, opacity, min font, online consent checkbox, one-time privacy warning.
- Produces: `RegionPicker(screen: Rect) -> list[Rect]` full-screen drag selection.

- [ ] **Step 1: Write failing hotkey test** for the binding parser: `parse_binding("Ctrl+Alt+Z") -> (MOD_CONTROL | MOD_ALT, ord("Z"))`; invalid string raises `ValueError`.
- [ ] **Step 2: Implement hotkeys and picker.**
- [ ] **Step 3: Implement settings dialog**; saving calls `save_config` and `pipeline.set_engine`.
- [ ] **Step 4: Wire `app.py`**: startup, tray-less, `Ctrl+Alt+Z/R/M/H/S`, overlay on top, shutdown on `Ctrl+Alt+Q`.
- [ ] **Step 5: Manual check** all hotkeys; commit `feat: add settings, hotkeys and region picker`.

---

### Task 14: HUD

**Files:**
- Create: `src/ocrtrans/hud.py`

**Interfaces:**
- Produces: `HudWindow(stats_provider: Callable[[], dict])` with `set_visible(bool)`; updates every 500 ms, always-on-top, click-through, `WDA_EXCLUDEFROMCAPTURE`; shows p50/p95, engine, health, `fit_truncated` count, `ocr-saturated`/`mt-saturated`.

- [ ] **Step 1: Implement** a small translucent text panel in the top-left; toggle with `Ctrl+Alt+H`.
- [ ] **Step 2: Manual check** values change while a dialogue plays; commit `feat: add performance HUD`.

---

### Task 15: Benchmark harness

**Files:**
- Create: `src/ocrtrans/bench/__main__.py`, `src/ocrtrans/bench/quality.py`, `src/ocrtrans/bench/perf.py`, `src/ocrtrans/bench/ocr_check.py`, `src/ocrtrans/bench/corpus/quality.jsonl`, `src/ocrtrans/bench/fixtures/*`, `tests/test_bench.py`

**Interfaces:**
- Produces: CLI `python -m ocrtrans.bench [--quality] [--perf] [--ocr] [--no-comet] [--engine all|<name>] [--out bench/report.md]`.
- Produces: `score_quality(pairs: list[tuple[str, str]]) -> dict` with `chrf_pp`, `comet`, `number_preserved`, `proper_noun_preserved`, `length_ratio`, `empty_count`.
- Produces: `run_replay(frames: list[Path], pipeline: Pipeline) -> dict` with per-stage and end-to-end `p50/p95/p99`, `ocr_calls`, `premature_rate`.

- [ ] **Step 1: Write corpus** — 80 lines across the 10 categories, each `{"id","category","text","ref_vi"}`; plus 30 rendered screenshots generated from them, and a 10 s animated clip.
- [ ] **Step 2: Write failing tests** for `number_preserved` (`"3 HP"` → preserved, `"ba HP"` → not) and percentile report shape.
- [ ] **Step 3: Implement** chrF++ and COMET via `sacrebleu`/`unbabel-comet` (optional import; `--no-comet` skips) and the replay harness that drives the real pipeline classes with fake capture frames.
- [ ] **Step 4: Run** `python -m ocrtrans.bench --engine all --perf --quality --no-comet`, confirm `bench/report.md` exists with one row per engine.
- [ ] **Step 5: Commit** `feat: add quality, perf and OCR benchmarks`.

---

### Task 16: Packaging and end-to-end check

**Files:**
- Create: `run.bat`, `README.md`, `tests/test_e2e_smoke.py`

**Interfaces:**
- Produces: `run.bat` that uses `.venv\Scripts\python.exe -m ocrtrans.app` and prints a clear error when the venv is missing.
- README: install, model download, hotkeys, engine table with free-tier limits, known limits (HDR, exclusive fullscreen, DRM, anti-cheat, API keys in plaintext).

- [ ] **Step 1: Implement `run.bat` and README.**
- [ ] **Step 2: Smoke test** `tests/test_e2e_smoke.py` runs the pipeline on 3 fixture frames with fake capture/OCR/engine and asserts overlay items are emitted for known text and cleared after 2 empty passes.
- [ ] **Step 3: Run the full suite** `.\.venv\Scripts\pytest -q` → all pass, GPU tests skipped unless `-m gpu`.
- [ ] **Step 4: Real-game manual check** borderless 1080p, dialogue-heavy game, both local engines, then one online engine; record p50/p95 from the HUD in the README table.
- [ ] **Step 5: Commit** `feat: add run script, docs and end-to-end smoke test`.
