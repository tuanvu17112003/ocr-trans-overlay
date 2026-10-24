"""Overlay lifecycle: paint a translation only while its exact source is on screen."""

from __future__ import annotations

from dataclasses import dataclass

from .capture import Health
from .geometry import Rect
from .pipeline import EventKind, PipelineEvent, RenderItem


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass
class _Block:
    block_id: int
    rect: Rect
    source: str = ""
    translated: str = ""
    translated_for: str = ""
    line_px: int = 0


class OverlayState:
    def __init__(
        self,
        unhealthy_hold_s: float = 1.0,
        watchdog_s: float = 1.5,
    ) -> None:
        self.unhealthy_hold_s = unhealthy_hold_s
        self.watchdog_s = watchdog_s
        self.hidden_stale = 0
        self._blocks: dict[int, _Block] = {}
        self._epoch = -1
        self._last_scan_id = 0
        self._last_ocr_at: float | None = None
        self._suppressed = False
        self._suppressed_at_scan = 0
        self._unhealthy_since: float | None = None

    def clear(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._blocks.clear()
        self.hidden_stale = 0
        self._last_scan_id = 0
        self._last_ocr_at = None
        self._suppressed = False
        self._suppressed_at_scan = 0
        self._unhealthy_since = None

    def state(self, block_id: int) -> str:
        block = self._blocks.get(block_id)
        if block is None or self._suppressed:
            return "hidden"
        if not block.translated:
            return "pending"
        return "shown"

    # ---- events -----------------------------------------------------------

    def on_event(self, event: PipelineEvent, now: float) -> list[RenderItem]:
        if event.kind is EventKind.RESET:
            self._reset()
            self._epoch = event.epoch
            return []

        if event.epoch < self._epoch:
            return self._render()

        self._epoch = max(self._epoch, event.epoch)

        if event.kind is EventKind.CAPTURE_HEALTH:
            self._on_health(event.health, now)
        elif event.kind is EventKind.OCR_RESULT:
            if event.scan_id <= self._last_scan_id:
                return self._render()  # stale completion, keep the screen
            self._on_ocr(event, now)
        elif event.kind is EventKind.TRANSLATION_RESULT:
            self._on_translation(event)

        self._watchdog(now)
        return self._render()

    def tick(self, now: float) -> list[RenderItem] | None:
        """Watchdog only; None means nothing changed."""
        was = self._suppressed
        self._watchdog(now)
        if self._suppressed == was:
            return None
        return self._render()

    def _on_health(self, health: Health | None, now: float) -> None:
        if health is Health.OK or health is Health.WGC_FALLBACK or health is None:
            self._unhealthy_since = None
            return
        if self._unhealthy_since is None:
            self._unhealthy_since = now
        if now - self._unhealthy_since >= self.unhealthy_hold_s:
            self._suppress()

    def _on_ocr(self, event: PipelineEvent, now: float) -> None:
        self._last_scan_id = event.scan_id
        self._last_ocr_at = now
        if event.scan_interval_s or event.scan_ms:
            gap = max(event.scan_interval_s, 1.5 * event.scan_ms / 1000.0)
            self.watchdog_s = max(1.5, gap + event.scan_ms / 1000.0 + 0.35)

        present: set[int] = set()
        for view in event.blocks:
            present.add(view.block_id)
            block = self._blocks.get(view.block_id)
            if block is None:
                block = _Block(block_id=view.block_id, rect=view.rect)
                self._blocks[view.block_id] = block
            if _norm(block.source) != _norm(view.source_text):
                block.source = view.source_text
                block.translated = ""
                block.translated_for = ""
            block.rect = view.rect
            block.line_px = view.line_px

        for block_id in [bid for bid in self._blocks if bid not in present]:
            del self._blocks[block_id]
            self.hidden_stale += 1

        if self._suppressed and event.scan_id > self._suppressed_at_scan:
            self._suppressed = False

    def _on_translation(self, event: PipelineEvent) -> None:
        for item in event.items:
            block = self._blocks.get(item.block_id)
            if block is None or _norm(block.source) != _norm(item.source_text):
                continue  # translation for text that is no longer on screen
            block.translated = item.text
            block.translated_for = item.source_text

    def _watchdog(self, now: float) -> None:
        if self._suppressed or self._last_ocr_at is None:
            return
        if now - self._last_ocr_at >= self.watchdog_s:
            self._suppress()

    def _suppress(self) -> None:
        if not self._suppressed:
            self._suppressed = True
            self._suppressed_at_scan = self._last_scan_id

    # ---- rendering --------------------------------------------------------

    def _render(self) -> list[RenderItem]:
        if self._suppressed:
            return []
        items: list[RenderItem] = []
        for block in self._blocks.values():
            if not block.translated:
                continue
            if _norm(block.translated_for) != _norm(block.source):
                continue
            items.append(
                RenderItem(
                    block_id=block.block_id,
                    text=block.translated,
                    rect=block.rect,
                    source_text=block.source,
                    cached=False,
                    line_px=block.line_px,
                )
            )
        return items
