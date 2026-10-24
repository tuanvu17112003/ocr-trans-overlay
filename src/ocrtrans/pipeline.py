"""Capture -> OCR -> translate pipeline: text-first scanning, bounded by a budget."""

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

from .blocks import BlockMemory, BlockState, TrackBlock
from .capture import CaptureManager, Health
from .exclusion import AffinityGate
from .geometry import Box, Rect, union_all
from .ocr import group_blocks
from .scan import MIN_INTERVAL_S, ScanBudget, frame_changed
from .stats import LatencyStats
from .translate import Engine, TranslationService

log = logging.getLogger(__name__)


def _offset(rect: Rect, origin: Rect) -> Rect:
    """Region-relative block rect -> screen rect."""
    return Rect(rect.x + origin.x, rect.y + origin.y, rect.w, rect.h)


class EventKind(str, Enum):
    OCR_RESULT = "ocr_result"
    TRANSLATION_RESULT = "translation_result"
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
    scan_id: int = 0
    blocks: list[BlockView] = field(default_factory=list)
    items: list[RenderItem] = field(default_factory=list)
    health: Health | None = None
    reason: str = ""
    scan_interval_s: float = 0.0
    scan_ms: float = 0.0


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
        service: TranslationService,
        emit: Callable[[PipelineEvent], None],
        *,
        memory: BlockMemory | None = None,
        budget: ScanBudget | None = None,
        gate: ExclusionGate | None = None,
        fps: int = 15,
        clock=time.monotonic,
    ) -> None:
        self.capture = capture
        self.ocr = ocr
        self.service = service
        self.emit = emit
        self.gate = gate
        self.fps = fps
        self.clock = clock
        self.memory = memory or BlockMemory()
        self.budget = budget or ScanBudget(fps)

        self.epoch = 0
        self.enabled = True
        self._regions: list[Rect] = []
        self._running = False
        self._threads: list[threading.Thread] = []
        self._ocr_queue = _Latest()
        self._mt_queue = _LatestPerKey()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="http")
        self._switch_lock = threading.Lock()
        self._last_mt: dict[int, tuple[str, int]] = {}

        self._scan_id = 0
        self._ocr_busy = threading.Event()
        self._wake_prev: np.ndarray | None = None
        self._interval = MIN_INTERVAL_S
        self._last_ocr_end = clock() - 10.0
        self._last_ocr_ms = 0.0
        self._empty_count = 0

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
        self.memory.reset()
        for thread in self._threads:
            thread.join(timeout)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _new_epoch(self) -> None:
        self.epoch += 1
        self._ocr_queue.clear()
        self._mt_queue.clear()
        self._last_mt.clear()
        self.memory.reset()
        self._scan_id = 0
        self._wake_prev = None
        self._empty_count = 0
        self._last_ocr_ms = 0.0
        self._interval = MIN_INTERVAL_S
        self._last_ocr_end = self.clock() - 10.0

    def set_regions(self, rects: list[Rect]) -> None:
        self._regions = list(rects)
        self._new_epoch()
        self.emit(PipelineEvent(kind=EventKind.RESET, epoch=self.epoch, at=self.clock()))

    def set_fps(self, fps: int) -> None:
        self.fps = fps
        self.budget.set_fps(fps)

    def set_profile(self, profile: str) -> None:
        self.memory.set_profile(profile)

    def unload_engine(self, timeout: float = 5.0) -> bool:
        """Free the current model. Only valid while paused; False means busy."""
        if self.enabled:
            raise RuntimeError("pause the pipeline before unloading the engine")
        with self._switch_lock:
            return self.service.release(timeout)

    def adopt_engine(self, engine: Engine) -> None:
        """Install an already-loaded engine after unload_engine()."""
        with self._switch_lock:
            self.service.adopt(engine)
        self._new_epoch()
        self.emit(PipelineEvent(kind=EventKind.RESET, epoch=self.epoch, at=self.clock()))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self._new_epoch()
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
            "mt_errors": self.service.errors,
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
            "ocr_saturated": self._ocr_busy.is_set(),
            "mt_saturated": self._mt_queue.qsize() > 0,
        }

    # ---- capture ----------------------------------------------------------

    @staticmethod
    def _views(blocks: list[TrackBlock], origin: Rect) -> list[BlockView]:
        return [
            BlockView(
                block_id=block.id,
                rect=_offset(block.rect, origin),
                source_text=block.text,
                state="stable" if block.state is BlockState.STABLE else "pending",
                line_px=block.line_px,
            )
            for block in blocks
        ]

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
            if self._ocr_busy.is_set():
                time.sleep(0.016)  # never replace a pending frame
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
                self.emit(
                    PipelineEvent(
                        kind=EventKind.CAPTURE_HEALTH,
                        epoch=self.epoch,
                        at=self.clock(),
                        health=health,
                    )
                )

            wake, self._wake_prev = frame_changed(self._wake_prev, frame)
            interval = self.budget.interval(self._last_ocr_ms, self._empty_count, wake)
            self._interval = interval
            if self.clock() - self._last_ocr_end < interval:
                self._sleep_rest(started)
                continue

            self._scan_id += 1
            self._ocr_queue.put((self.epoch, self._scan_id, frame, self._region()))
            self._sleep_rest(started)

    # ---- OCR --------------------------------------------------------------

    def _ocr_loop(self) -> None:
        while self._running:
            task = self._ocr_queue.take()
            if task is None:
                continue
            epoch, scan_id, frame, origin = task
            if epoch != self.epoch or not self.enabled:
                continue
            self._ocr_busy.set()
            started = self.clock()
            try:
                blocks = self._run_ocr(frame)
            except Exception:  # noqa: BLE001 - OCR failure must not kill the loop
                self._ocr_errors += 1
                if self._ocr_errors == 1 or self._ocr_errors % 100 == 0:
                    log.exception("OCR pass failed (%d so far)", self._ocr_errors)
                continue
            finally:
                self._last_ocr_end = self.clock()
                self._last_ocr_ms = (self._last_ocr_end - started) * 1000
                self._ocr_busy.clear()
            if epoch != self.epoch:
                continue
            self._ocr_stats.add(self._last_ocr_ms, cached=False)
            self._ocr_passes += 1
            self._last_blocks = len(blocks)
            self._empty_count = 0 if blocks else self._empty_count + 1
            self.emit(
                PipelineEvent(
                    kind=EventKind.OCR_RESULT,
                    epoch=epoch,
                    at=self.clock(),
                    scan_id=scan_id,
                    blocks=self._views(blocks, origin),
                    scan_interval_s=self._interval,
                    scan_ms=self._last_ocr_ms,
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

    def _run_ocr(self, frame: np.ndarray) -> list[TrackBlock]:
        grouped = group_blocks(self.ocr.read(frame))
        boxes = [
            Box(
                rect=block.rect,
                text=block.text,
                confidence=min(line.confidence for line in block.lines),
                line_px=block.line_px,
            )
            for block in grouped
        ]
        ttl = self.budget.ttl(self._interval)
        return self.memory.observe(boxes, self.clock(), ttl_s=ttl)

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
        if not self.enabled or epoch != self.epoch:
            return
        self._inflight += 1
        started = self.clock()
        try:
            result, cached = self.service.translate(text)
            self._mt_stats.add((self.clock() - started) * 1000, cached=cached)
        except Exception:  # noqa: BLE001 - provider errors are degraded, not fatal
            return
        finally:
            self._inflight -= 1
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
