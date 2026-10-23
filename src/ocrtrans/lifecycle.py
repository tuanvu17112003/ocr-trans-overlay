"""Overlay lifecycle: which translated blocks stay on screen, and when they go."""

from __future__ import annotations

from dataclasses import dataclass

from .capture import Health
from .pipeline import EventKind, PipelineEvent, RenderItem


@dataclass
class _Block:
    block_id: int
    rect: object
    source: str = ""
    translated: str = ""
    miss: int = 0
    stale_since: float | None = None
    state: str = "shown"


class OverlayState:
    def __init__(
        self,
        stale_cap_s: float = 1.0,
        miss_passes: int = 2,
        unhealthy_hold_s: float = 1.0,
    ) -> None:
        self.stale_cap_s = stale_cap_s
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
        if block.stale_since is not None:
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
            block = self._blocks.get(view.block_id)
            if block is None:
                block = _Block(block_id=view.block_id, rect=view.rect)
                self._blocks[view.block_id] = block
            elif block.source != view.source_text:
                block.stale_since = now
            block.rect = view.rect
            block.source = view.source_text
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
            if block is None or block.source != item.source_text:
                continue  # translation for text that is no longer on screen
            block.translated = item.text
            block.stale_since = None
            block.state = "shown"

    # ---- rendering --------------------------------------------------------

    def _render(self, now: float) -> list[RenderItem]:
        items: list[RenderItem] = []
        for block in self._blocks.values():
            if block.state == "hidden":
                continue
            if block.stale_since is not None and now - block.stale_since >= self.stale_cap_s:
                block.state = "hidden"
                continue
            items.append(
                RenderItem(
                    block_id=block.block_id,
                    text=block.translated or block.source,
                    rect=block.rect,
                    source_text=block.source,
                    cached=False,
                )
            )
        return items
