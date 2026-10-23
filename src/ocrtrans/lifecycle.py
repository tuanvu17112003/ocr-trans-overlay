"""Overlay lifecycle: which translated blocks stay on screen, and when they go."""

from __future__ import annotations

from dataclasses import dataclass

from .capture import Health
from .pipeline import EventKind, PipelineEvent, RenderItem


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class _Block:
    block_id: int
    rect: object
    source: str = ""
    translated: str = ""
    miss: int = 0
    state: str = "shown"
    line_px: int = 0


class OverlayState:
    def __init__(
        self,
        miss_passes: int = 2,
        unhealthy_hold_s: float = 1.0,
    ) -> None:
        self.miss_passes = miss_passes
        self.unhealthy_hold_s = unhealthy_hold_s
        self._blocks: dict[int, _Block] = {}
        self._epoch = -1
        self._unhealthy_since: float | None = None

    def clear(self) -> None:
        self._blocks.clear()
        self._unhealthy_since = None

    def state(self, block_id: int) -> str:
        block = self._blocks.get(block_id)
        if block is None:
            return "hidden"
        if block.state == "hidden":
            return "hidden"
        if not block.translated:
            return "pending"
        return "shown"

    # ---- events -----------------------------------------------------------

    def on_event(self, event: PipelineEvent, now: float) -> list[RenderItem]:
        if event.kind is EventKind.RESET:
            self.clear()
            self._epoch = event.epoch
            return []

        if event.epoch < self._epoch:
            return self._render(now)  # ignore, but keep the screen as it is

        self._epoch = max(self._epoch, event.epoch)

        if event.kind is EventKind.CAPTURE_HEALTH:
            return self._on_health(event.health, now)
        if event.kind is EventKind.OCR_RESULT:
            self._on_ocr(event, now)
        elif event.kind is EventKind.TRANSLATION_RESULT:
            self._on_translation(event)
        # SKIP changes nothing

        return self._render(now)

    def _on_health(self, health: Health | None, now: float) -> list[RenderItem]:
        if health is Health.OK or health is None:
            self._unhealthy_since = None
            return self._render(now)
        if self._unhealthy_since is None:
            self._unhealthy_since = now
        if now - self._unhealthy_since >= self.unhealthy_hold_s:
            for block in self._blocks.values():
                block.state = "hidden"
            return []
        return self._render(now)

    def _on_ocr(self, event: PipelineEvent, now: float) -> None:
        present = {view.block_id for view in event.blocks}
        for view in event.blocks:
            if view.state != "stable":
                block = self._blocks.get(view.block_id)
                if block is not None and _norm(block.source) != _norm(view.source_text):
                    block.translated = ""
                continue
            block = self._blocks.get(view.block_id)
            if block is None:
                block = _Block(block_id=view.block_id, rect=view.rect)
                self._blocks[view.block_id] = block
            elif _norm(block.source) != _norm(view.source_text):
                block.translated = ""
            block.rect = view.rect
            block.source = view.source_text
            block.line_px = view.line_px
            block.miss = 0
            block.state = "shown"

        observed = event.observed_block_ids
        scope = set(self._blocks) if observed is None else set(observed)
        for block_id in scope:
            if block_id in present:
                continue
            block = self._blocks.get(block_id)
            if block is None:
                continue
            block.miss += 1
            if block.miss >= self.miss_passes:
                block.state = "hidden"

    def _on_translation(self, event: PipelineEvent) -> None:
        for item in event.items:
            block = self._blocks.get(item.block_id)
            if block is None or _norm(block.source) != _norm(item.source_text):
                continue  # translation for text that is no longer on screen
            block.translated = item.text
            block.state = "shown"

    # ---- rendering --------------------------------------------------------

    def _render(self, now: float) -> list[RenderItem]:
        items: list[RenderItem] = []
        for block in self._blocks.values():
            if block.state == "hidden" or not block.translated:
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

