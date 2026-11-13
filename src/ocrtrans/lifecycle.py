"""Overlay lifecycle: paint a translation only while its exact source is on screen."""

from __future__ import annotations

from dataclasses import dataclass

from .capture import Health
from .geometry import Rect
from .pipeline import EventKind, PipelineEvent, RenderItem
from .textnorm import change_class, compare_key


def _norm(text: str) -> str:
    return compare_key(text)


@dataclass
class _Block:
    block_id: int
    rect: Rect
    source: str = ""
    translated: str = ""
    translated_for: str = ""
    line_px: int = 0
    # Text the next translation should target while the previous cover is still
    # up. Set when a cosmetic wobble arrives.
    pending_source: str = ""
    # Wall clock after which a held cover must be dropped even without a new
    # translation, so a stalled translator cannot pin the wrong text.
    hold_until: float = 0.0


class OverlayState:
    """Turns pipeline events into the list of blocks to paint.

    The one rule that outranks everything: never paint a translation over text it
    was not requested for. The previous build satisfied that rule with an exact
    comparison, which also meant one character of OCR noise tore the cover down
    and rebuilt it a scan later -- the visible strobing.

    So a wobble is classified before it acts (textnorm.change_class):

      same      spacing, trailing punctuation, confusable glyphs: do nothing.
      cosmetic  a typo-scale difference in a long line: keep the cover up and
                retranslate quietly, up to hold_max_s. The stored source string is
                not updated until the new translation lands, so the invariant
                translated_for == source still holds while it is held.
      semantic  anything meaningful, and always a digit change: drop the
                translation immediately. Hiding is acceptable, being wrong is not.
    """

    def __init__(
        self,
        unhealthy_hold_s: float = 1.0,
        watchdog_s: float = 1.5,
        hold_max_s: float = 0.8,
    ) -> None:
        self.unhealthy_hold_s = unhealthy_hold_s
        self.watchdog_s = watchdog_s
        self.hold_max_s = hold_max_s
        self.hidden_stale = 0
        # Covers that were kept through a wobble, for the stats line.
        self.held = 0
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
        self.held = 0
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

    def known_rects(self) -> dict[int, Rect]:
        """Every tracked block, translated or not.

        The overlay uses this to decide how far a grown cover may extend: a block
        with no translation yet still owns the space under it, so a neighbour is
        not re-laid out when it appears.
        """
        return {block_id: block.rect for block_id, block in self._blocks.items()}

    # ---- events -----------------------------------------------------------

    def on_event(self, event: PipelineEvent, now: float) -> list[RenderItem]:
        if event.kind is EventKind.RESET:
            self._reset()
            self._epoch = event.epoch
            return []

        if event.epoch < self._epoch:
            return self._render(now)

        self._epoch = max(self._epoch, event.epoch)

        if event.kind is EventKind.CAPTURE_HEALTH:
            self._on_health(event.health, now)
        elif event.kind is EventKind.OCR_RESULT:
            if event.scan_id <= self._last_scan_id:
                return self._render(now)
            self._on_ocr(event, now)
        elif event.kind is EventKind.TRANSLATION_RESULT:
            self._on_translation(event)

        self._watchdog(now)
        return self._render(now)

    def tick(self, now: float) -> list[RenderItem] | None:
        """Watchdog and hold expiry; None means nothing changed.

        A hold expiring is a render change even though the block dict is
        untouched: the item list shrinks. Comparing the rendered count is what
        makes the expiry reach the screen.
        """
        was = (self._suppressed, self._held_count(now), len(self._render(now)))
        self._expire_holds(now)
        self._watchdog(now)
        if was == (self._suppressed, self._held_count(now), len(self._render(now))):
            return None
        return self._render(now)

    def _expire_holds(self, now: float) -> None:
        """Adopt the pending text once a hold runs out.

        Adopting is what hides the cover: `translated_for` still names the old
        text, so the render check drops the block. Without this the cover would
        stay up forever with a translation for text that is no longer there.
        """
        for block in self._blocks.values():
            if block.pending_source and block.hold_until <= now:
                block.source = block.pending_source
                block.pending_source = ""
                block.hold_until = 0.0

    def _held_count(self, now: float) -> int:
        return sum(1 for block in self._blocks.values() if self._is_held(block, now))

    @staticmethod
    def _is_held(block: _Block, now: float) -> bool:
        return (
            bool(block.translated)
            and block.hold_until > now
            and bool(block.pending_source)
        )

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
                block = _Block(
                    block_id=view.block_id,
                    rect=view.rect,
                    source=view.source_text,
                    line_px=view.line_px,
                )
                self._blocks[view.block_id] = block
                continue
            self._apply_text(block, view.source_text, now)
            block.rect = view.rect
            block.line_px = view.line_px

        for block_id in [bid for bid in self._blocks if bid not in present]:
            del self._blocks[block_id]
            self.hidden_stale += 1

        if self._suppressed and event.scan_id > self._suppressed_at_scan:
            self._suppressed = False

    def _apply_text(self, block: _Block, new_text: str, now: float) -> None:
        """Decide what a new reading means for the cover that is already up."""
        if not block.translated:
            block.source = new_text
            block.pending_source = ""
            return

        kind = change_class(block.source, new_text)
        if kind == "same":
            # Nothing meaningful changed. Keep the exact stored source so that
            # translated_for == source still holds and nothing is repainted.
            return
        if kind == "cosmetic":
            # Hold the current cover, aim the next translation at the new text.
            if not block.pending_source:
                block.hold_until = now + self.hold_max_s
                self.held += 1
            block.pending_source = new_text
            return
        # semantic: drop it now.
        block.source = new_text
        block.translated = ""
        block.translated_for = ""
        block.pending_source = ""
        block.hold_until = 0.0

    def _on_translation(self, event: PipelineEvent) -> None:
        for item in event.items:
            block = self._blocks.get(item.block_id)
            if block is None:
                continue
            wanted = block.pending_source or block.source
            if _norm(item.source_text) != _norm(wanted):
                continue  # a translation for text that is no longer on screen
            block.translated = item.text
            block.translated_for = item.source_text
            # The swap is atomic: the source only changes now, so a held cover is
            # valid before and after the replacement.
            block.source = item.source_text
            block.pending_source = ""
            block.hold_until = 0.0

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

    def _render(self, now: float) -> list[RenderItem]:
        if self._suppressed:
            return []
        items: list[RenderItem] = []
        for block in self._blocks.values():
            if not block.translated:
                continue
            held = self._is_held(block, now)
            if not held and _norm(block.translated_for) != _norm(block.source):
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