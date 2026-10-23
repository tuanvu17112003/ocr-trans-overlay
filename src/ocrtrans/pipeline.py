"""Capture -> OCR -> translate pipeline with epoch cancellation."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Protocol

import numpy as np

from .capture import CaptureManager, Health
from .exclusion import AffinityGate
from .geometry import Box, Rect, union_all
from .ocr import group_blocks
from .stats import LatencyStats
from .track import BlockState, Decision, TemporalTracker, TrackBlock
from .translate import Engine, TranslationCache, TranslationService

log = logging.getLogger(__name__)


def _offset(rect: Rect, origin: Rect) -> Rect:
    """Region-relative tracker rect -> screen rect."""
    return Rect(rect.x + origin.x, rect.y + origin.y, rect.w, rect.h)


class EventKind(str, Enum):
    OCR_RESULT = "ocr_result"
    TRANSLATION_RESULT = "translation_result"
    SKIP = "skip"
    CAPTURE_HEALTH = "capture_health"
    RESET = "reset"


@dataclass
class BlockView:
    block_id: int
    rect: Rect
    source_text: str
    state: Literal["pending", "stable"]
    line_px: int = 0


@dataclass
class RenderItem:
    block_id: int
    text: str
    rect: Rect
    source_text: str
    cached: bool
    line_px: int = 0


@dataclass
class PipelineEvent:
    kind: EventKind
    epoch: int
    at: float
    blocks: list[BlockView] = field(default_factory=list)
    observed_block_ids: list[int] = field(default_factory=list)
    items: list[RenderItem] = field(default_factory=list)
    health: Health | None = None
    reason: str = ""


class ExclusionGate(Protocol):
    def acquire(self, deadline_ms: int = 8) -> bool: ...

    def release(self) -> None: ...


class _Latest:
    """Depth-1 newest-wins slot."""

    def __init__(self) -> None:
        self._item = None
        self._cv = threading.Condition()

    def put(self, item) -> None:
        with self._cv:
            self._item = item
            self._cv.notify()

    def take(self, timeout: float = 0.05):
        with self._cv:
            if self._item is None:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item

    def clear(self) -> None:
        with self._cv:
            self._item = None

    def qsize(self) -> int:
        with self._cv:
            return 0 if self._item is None else 1


class _LatestPerKey:
    """Depth-1 newest-wins slot per block id."""

    def __init__(self) -> None:
        self._items: OrderedDict[int, object] = OrderedDict()
        self._cv = threading.Condition()

    def put(self, key: int, item) -> None:
        with self._cv:
            self._items[key] = item
            self._items.move_to_end(key)
            self._cv.notify()

    def take(self, timeout: float = 0.05):
        with self._cv:
            if not self._items:
                self._cv.wait(timeout)
            if not self._items:
                return None
            return self._items.popitem(last=False)

    def clear(self) -> None:
        with self._cv:
            self._items.clear()

    def qsize(self) -> int:
        with self._cv:
            return len(self._items)


class Pipeline:
    def __init__(
        self,
        capture: CaptureManager,
        ocr,
        tracker: TemporalTracker,
        service: TranslationService,
        emit: Callable[[PipelineEvent], None],
        gate: ExclusionGate | None = None,
        fps: int = 15,
        clock=time.monotonic,
    ) -> None:
        self.capture = capture
        self.ocr = ocr
        self.tracker = tracker
        self.service = service
        self.emit = emit
        self.gate = gate
        self.fps = fps
        self.clock = clock

        self.epoch = 0
        self.enabled = True
        self._regions: list[Rect] = []
        self._running = False
        self._threads: list[threading.Thread] = []
        self._ocr_queue = _Latest()
        self._mt_queue = _LatestPerKey()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="http")
        self._pending_engine: Engine | None = None
        self._switch_lock = threading.Lock()
        self._last_mt: dict[int, tuple[str, int]] = {}
        self._known_ids: set[int] = set()

        self._ocr_passes = 0
        self._ocr_errors = 0
        self._last_blocks = 0
        self._exclusion_missed = 0
        self._fit_truncated = 0
        self._inflight = 0
        self._ocr_stats = LatencyStats()
        self._mt_stats = LatencyStats()

    # ---- public API -------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._ocr_loop, name="ocr", daemon=True),
            threading.Thread(target=self._mt_loop, name="mt", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if not self._running:
            return
        self._running = False
        self.epoch += 1
        self._ocr_queue.clear()
        self._mt_queue.clear()
        self._known_ids.clear()
        for thread in self._threads:
            thread.join(timeout)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def set_regions(self, rects: list[Rect]) -> None:
        self._regions = list(rects)
        self.epoch += 1
        self._ocr_queue.clear()
        self._mt_queue.clear()
        self._last_mt.clear()
        self._known_ids.clear()
        self.tracker.reset()
        self.emit(PipelineEvent(kind=EventKind.RESET, epoch=self.epoch, at=self.clock()))

    def set_engine(self, engine: Engine) -> None:
        self.epoch += 1
        self._pending_engine = engine
        self._mt_queue.clear()
        self._last_mt.clear()
        self.emit(PipelineEvent(kind=EventKind.RESET, epoch=self.epoch, at=self.clock()))

    def unload_engine(self) -> None:
        """Free the current model. Only valid while paused."""
        if self.enabled:
            raise RuntimeError("pause the pipeline before unloading the engine")
        with self._switch_lock:
            self.service.engine.unload()
            self._pending_engine = None

    def adopt_engine(self, engine: Engine) -> None:
        """Install an already-loaded engine after unload_engine()."""
        with self._switch_lock:
            old = self.service.engine
            if old is not engine:
                old.unload()
            self.service.engine = engine
            self.service.cache = TranslationCache()
            self._pending_engine = None
        self.epoch += 1
        self._ocr_queue.clear()
        self._mt_queue.clear()
        self._last_mt.clear()
        self._known_ids.clear()
        self.tracker.reset()
        self.emit(PipelineEvent(kind=EventKind.RESET, epoch=self.epoch, at=self.clock()))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.epoch += 1
        self._ocr_queue.clear()
        self._mt_queue.clear()
        self._known_ids.clear()
        self.emit(
            PipelineEvent(
                kind=EventKind.RESET,
                epoch=self.epoch,
                at=self.clock(),
                reason="enabled" if enabled else "disabled",
            )
        )

    def stats(self) -> dict:
        return {
            "ocr_passes": self._ocr_passes,
            "ocr_errors": self._ocr_errors,
            "blocks": self._last_blocks,
            "ocr_queue": self._ocr_queue.qsize(),
            "mt_queue": self._mt_queue.qsize(),
            "inflight": self._inflight,
            "epoch": self.epoch,
            "fit_truncated": self._fit_truncated,
            "exclusion_missed": self._exclusion_missed,
            "p50": round(self._mt_stats.percentile(50), 1),
            "p95": round(self._mt_stats.percentile(95), 1),
            "ocr_p50": round(self._ocr_stats.percentile(50), 1),
            "ocr_p95": round(self._ocr_stats.percentile(95), 1),
            "ocr_saturated": self._ocr_queue.qsize() > 0,
            "mt_saturated": self._mt_queue.qsize() > 0,
        }

    # ---- capture ----------------------------------------------------------

    def _region(self) -> Rect:
        return union_all(self._regions) if self._regions else Rect(0, 0, 1920, 1080)

    def _sleep_rest(self, started: float) -> None:
        remaining = (1.0 / self.fps) - (self.clock() - started)
        if remaining > 0:
            time.sleep(remaining)

    def _capture_loop(self) -> None:
        while self._running:
            started = self.clock()
            if not self.enabled:
                time.sleep(0.05)
                continue
            gate = self.gate
            if gate is not None and not gate.acquire():
                self._exclusion_missed += 1
                self._sleep_rest(started)
                continue
            try:
                frame, health = self.capture.grab(self._region())
            finally:
                if gate is not None:
                    gate.release()
            if frame is None:
                self._sleep_rest(started)
                continue
            if health is not Health.OK:
                self.tracker.reset()
                self.emit(
                    PipelineEvent(
                        kind=EventKind.CAPTURE_HEALTH,
                        epoch=self.epoch,
                        at=self.clock(),
                        health=health,
                    )
                )
                self._sleep_rest(started)
                continue

            decision = self.tracker.update(frame, self.clock())
            if decision.skip:
                self.emit(PipelineEvent(kind=EventKind.SKIP, epoch=self.epoch, at=self.clock()))
            else:
                self._ocr_queue.put((self.epoch, frame, decision, self._region()))
            self._sleep_rest(started)

    # ---- OCR --------------------------------------------------------------

    def _ocr_loop(self) -> None:
        while self._running:
            task = self._ocr_queue.take()
            if task is None:
                continue
            epoch, frame, decision, origin = task
            if epoch != self.epoch or not self.enabled:
                continue
            try:
                started = self.clock()
                blocks, observed = self._run_ocr(frame, decision)
                self._ocr_stats.add((self.clock() - started) * 1000, cached=False)
            except Exception:  # noqa: BLE001 - OCR failure must not kill the loop
                self._ocr_errors += 1
                if self._ocr_errors == 1 or self._ocr_errors % 100 == 0:
                    log.exception("OCR pass failed (%d so far)", self._ocr_errors)
                continue
            if epoch != self.epoch:
                continue
            self._ocr_passes += 1
            self._last_blocks = len(blocks)
            views = [
                BlockView(
                    block_id=block.id,
                    rect=_offset(block.rect, origin),
                    source_text=block.text,
                    state="stable" if block.state is BlockState.STABLE else "pending",
                    line_px=block.line_px,
                )
                for block in blocks
            ]
            self.emit(
                PipelineEvent(
                    kind=EventKind.OCR_RESULT,
                    epoch=epoch,
                    at=self.clock(),
                    blocks=views,
                    observed_block_ids=observed,
                )
            )
            candidates = [
                block
                for block in blocks
                if block.state is BlockState.STABLE
                and len(block.text.strip()) >= 3
                and self._last_mt.get(block.id) != (block.text, epoch)
            ]
            candidates.sort(key=lambda block: block.rect.w * block.rect.h, reverse=True)
            for block in candidates[:12]:
                self._submit_mt(block, epoch, origin)

    def _run_ocr(self, frame: np.ndarray, decision: Decision) -> tuple[list[TrackBlock], list[int]]:
        if decision.full_detect:
            known = set(self._known_ids)
            boxes = self.ocr.read(frame)
            grouped = group_blocks(boxes)
            boxes = [
                Box(
                    rect=block.rect,
                    text=block.text,
                    confidence=min(line.confidence for line in block.lines),
                    line_px=block.line_px,
                )
                for block in grouped
            ]
            blocks = self.tracker.observe(boxes, self.clock())
            observed = sorted(known | {block.id for block in blocks})
        else:
            rects = [rect for _, rect in decision.crops]
            boxes = self.ocr.read(frame, rects)
            observed = [block_id for block_id, _ in decision.crops]
            blocks = self.tracker.observe(boxes, self.clock(), scope=set(observed))
        self._known_ids = set(observed)
        seen = set(observed)
        return [block for block in blocks if block.id in seen], observed

    # ---- translation ------------------------------------------------------

    def _submit_mt(self, block: TrackBlock, epoch: int, origin: Rect) -> None:
        key = (block.text, epoch)
        if self._last_mt.get(block.id) == key:
            return
        self._last_mt[block.id] = key
        self._mt_queue.put(
            block.id, (epoch, block.text, _offset(block.rect, origin), block.line_px)
        )

    def _mt_loop(self) -> None:
        while self._running:
            self._maybe_switch()
            task = self._mt_queue.take()
            if task is None:
                continue
            tasks = [task]
            while len(tasks) < 8:
                extra = self._mt_queue.take(timeout=0)
                if extra is None:
                    break
                tasks.append(extra)
            current = [
                (block_id, payload)
                for block_id, payload in tasks
                if payload[0] == self.epoch and self.enabled
            ]
            if not current:
                continue
            if getattr(self.service.engine, "kind", "local") == "online":
                for block_id, (epoch, text, rect, line_px) in current:
                    self._pool.submit(self._translate, block_id, epoch, text, rect, line_px)
            else:
                self._translate_batch(current)

    def _translate_batch(self, tasks: list[tuple[int, tuple]]) -> None:
        started = self.clock()
        self._inflight += len(tasks)
        try:
            results = self.service.translate_many([payload[1] for _, payload in tasks])
        except Exception:  # noqa: BLE001 - provider errors must not kill the loop
            for block_id, _ in tasks:
                self._last_mt.pop(block_id, None)
            return
        finally:
            self._inflight -= len(tasks)
        self._mt_stats.add((self.clock() - started) * 1000, cached=False)

        items: list[RenderItem] = []
        for (block_id, (epoch, text, rect, line_px)), (result, cached) in zip(tasks, results):
            if epoch != self.epoch or not self.enabled:
                continue
            if not result:
                self._last_mt.pop(block_id, None)
                continue
            items.append(
                RenderItem(
                    block_id=block_id,
                    text=result,
                    rect=rect,
                    source_text=text,
                    cached=cached,
                    line_px=line_px,
                )
            )
        self._maybe_switch()
        if items:
            self.emit(
                PipelineEvent(
                    kind=EventKind.TRANSLATION_RESULT,
                    epoch=self.epoch,
                    at=self.clock(),
                    items=items,
                )
            )

    def _translate(self, block_id: int, epoch: int, text: str, rect: Rect, line_px: int) -> None:
        self._inflight += 1
        started = self.clock()
        try:
            result, cached = self.service.translate(text)
            self._mt_stats.add((self.clock() - started) * 1000, cached=cached)
        except Exception:  # noqa: BLE001 - provider errors are degraded, not fatal
            return
        finally:
            self._inflight -= 1
        self._maybe_switch()
        if epoch != self.epoch or not self.enabled:
            return
        self.emit(
            PipelineEvent(
                kind=EventKind.TRANSLATION_RESULT,
                epoch=epoch,
                at=self.clock(),
                items=[
                    RenderItem(
                        block_id=block_id,
                        text=result,
                        rect=rect,
                        source_text=text,
                        cached=cached,
                        line_px=line_px,
                    )
                ],
            )
        )

    def _maybe_switch(self) -> None:
        with self._switch_lock:
            engine = self._pending_engine
            if engine is None:
                return
            try:
                self.service.switch(engine)
            except RuntimeError:
                return  # still translating; retry on the next loop tick
            except Exception:  # noqa: BLE001 - a broken model must not kill the MT thread
                log.exception("engine switch failed")
                self._pending_engine = None
                return
            self._pending_engine = None
            # translations are engine-specific
            self.service.cache = TranslationCache()

