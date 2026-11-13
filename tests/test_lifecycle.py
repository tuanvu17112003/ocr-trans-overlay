from ocrtrans.capture import Health
from ocrtrans.geometry import Rect
from ocrtrans.lifecycle import OverlayState
from ocrtrans.pipeline import BlockView, EventKind, PipelineEvent, RenderItem


def item(block_id, text, source):
    return RenderItem(block_id, text, Rect(0, 0, 10, 10), source, False, 0)


def block(block_id, text, state="stable"):
    return BlockView(block_id, Rect(0, 0, 10, 10), text, state, 18)


def ev(kind, epoch=1, scan_id=1, **kw):
    return PipelineEvent(kind=kind, epoch=epoch, at=0.0, scan_id=scan_id, **kw)


def shown_state(text="HELLO", translation="XIN CHÀO"):
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, text)]), 0.0)
    s.on_event(
        ev(EventKind.TRANSLATION_RESULT, scan_id=1, items=[item(1, translation, text)]),
        0.1,
    )
    return s


def test_a_block_that_left_the_snapshot_disappears_at_once():
    s = shown_state()
    assert [
        i.text for i in s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[]), 0.2)
    ] == []


def test_digit_change_never_keeps_the_old_translation():
    s = shown_state("You have 100 HP", "Bạn có 100 HP")
    assert (
        s.on_event(
            ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "You have 10 HP")]),
            0.2,
        )
        == []
    )


def test_translation_for_old_source_is_dropped():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "NEW TEXT")]), 0.0)
    out = s.on_event(
        ev(EventKind.TRANSLATION_RESULT, scan_id=1, items=[item(1, "CŨ", "OLD TEXT")]),
        0.1,
    )
    assert out == []


def test_stale_snapshot_is_ignored():
    s = shown_state("NEW", "MỚI")
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "NEW")]), 0.2)
    out = s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, "OLD")]), 0.3)
    assert [i.text for i in out] == ["MỚI"]


def test_fallback_capture_keeps_painting():
    s = shown_state()
    out = s.on_event(
        ev(EventKind.CAPTURE_HEALTH, health=Health.WGC_FALLBACK), 0.5
    )
    assert [i.text for i in out] == ["XIN CHÀO"]


def test_black_capture_hides_after_one_second():
    s = shown_state()
    s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.BLACK), 0.5)
    assert s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.BLACK), 1.6) == []


def test_health_recovery_alone_does_not_repaint():
    s = shown_state()
    s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.BLACK), 0.5)
    s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.BLACK), 1.6)
    assert s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.OK), 1.7) == []


def test_watchdog_suppresses_then_repaints_on_a_fresh_snapshot_without_new_translation():
    s = OverlayState(watchdog_s=1.5)
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, "HELLO")]), 0.0)
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHÀO", "HELLO")]), 0.1)
    assert s.tick(1.6) == []
    assert s.tick(1.7) is None
    out = s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "HELLO")]), 1.8)
    assert [i.text for i in out] == ["XIN CHÀO"]


def test_watchdog_scales_with_a_slow_pass_cycle():
    s = OverlayState()
    s.on_event(
        ev(
            EventKind.OCR_RESULT,
            scan_id=1,
            scan_interval_s=1.278,
            scan_ms=852,
            blocks=[block(1, "HELLO")],
        ),
        0.0,
    )
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHÀO", "HELLO")]), 0.1)
    assert s.tick(2.0) is None          # watchdog = 1.278 + 0.852 + 0.35 = 2.48 s
    assert s.tick(2.5) == []


def test_watchdog_uses_the_gap_after_this_pass_not_the_one_before():
    s = OverlayState()
    s.on_event(
        ev(
            EventKind.OCR_RESULT,
            scan_id=1,
            scan_interval_s=0.25,   # the pass before this one was fast
            scan_ms=852,            # but this one was slow
            blocks=[block(1, "HELLO")],
        ),
        0.0,
    )
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHÀO", "HELLO")]), 0.1)
    assert s.tick(1.6) is None          # the next start is 1.28 s after this pass, not 0.25 s
    assert s.tick(2.5) == []


def test_stale_cover_does_not_return_after_a_capture_gap():
    s = OverlayState(watchdog_s=1.5)
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, "A")]), 0.0)
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "A-VI", "A")]), 0.1)
    assert s.tick(1.6) == []
    s.on_event(ev(EventKind.CAPTURE_HEALTH, health=Health.OK), 1.7)
    assert s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "B")]), 1.8) == []
    assert s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "A-VI", "A")]), 1.9) == []


def test_reset_clears_and_older_epoch_is_ignored():
    s = shown_state("NEW", "MỚI")
    assert s.on_event(ev(EventKind.RESET, epoch=2), 0.3) == []
    assert s.state(1) == "hidden"
    out = s.on_event(
        ev(EventKind.OCR_RESULT, epoch=1, scan_id=1, blocks=[block(1, "OLD")]), 0.4
    )
    assert out == []


def test_reset_accepts_the_first_snapshot_of_the_new_epoch():
    s = OverlayState()
    for scan in range(1, 5):
        s.on_event(
            ev(EventKind.OCR_RESULT, scan_id=scan, blocks=[block(1, "HELLO")]),
            0.1 * scan,
        )
    s.on_event(ev(EventKind.RESET, epoch=2), 0.5)
    s.on_event(ev(EventKind.OCR_RESULT, epoch=2, scan_id=1, blocks=[block(1, "HELLO")]), 0.6)
    out = s.on_event(
        ev(
            EventKind.TRANSLATION_RESULT,
            epoch=2,
            scan_id=1,
            items=[item(1, "XIN CHÀO", "HELLO")],
        ),
        0.7,
    )
    assert [i.text for i in out] == ["XIN CHÀO"]


def test_a_removed_block_is_counted_as_hidden_stale():
    s = shown_state()
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[]), 0.2)
    assert s.hidden_stale == 1


def test_render_keeps_source_line_height():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, "HELLO")]), 0.0)
    s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHÀO", "HELLO")]), 0.1)
    out = s.on_event(ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "HELLO")]), 0.2)
    assert out[0].line_px == 18


def test_pending_block_is_not_rendered():
    s = OverlayState()
    out = s.on_event(
        ev(EventKind.OCR_RESULT, blocks=[block(1, "HEL", state="pending")]), 0.0
    )
    assert out == []

# ---- cosmetic hold: the anti-strobe behaviour -----------------------------


def test_a_cosmetic_wobble_keeps_the_cover_up():
    """One OCR character dropping out of a long line must not tear the cover down."""
    s = shown_state("Open the gate, traveler.", "Mở cánh cổng, lữ khách.")
    out = s.on_event(
        ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "Open the gate traveler")]),
        0.2,
    )
    assert [i.text for i in out] == ["Mở cánh cổng, lữ khách."]
    assert s.held == 1


def test_the_held_cover_is_dropped_when_the_hold_expires():
    s = OverlayState(hold_max_s=0.8)
    s.on_event(ev(EventKind.OCR_RESULT, scan_id=1, blocks=[block(1, "Open the gate, traveler.")]), 0.0)
    s.on_event(
        ev(EventKind.TRANSLATION_RESULT, items=[item(1, "MỞ CỔNG", "Open the gate, traveler.")]),
        0.1,
    )
    assert s.on_event(
        ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "Open the gate traveler")]), 0.2
    )
    # Still inside the hold window.
    assert s.tick(0.5) is None
    # Past it: the cover must go rather than pin a translation for text that has
    # since changed.
    assert s.tick(1.2) == []


def test_a_held_cover_is_replaced_atomically_by_the_new_translation():
    s = shown_state("Open the gate, traveler.", "MỞ CỔNG")
    s.on_event(
        ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "Open the gate traveler")]), 0.2
    )
    out = s.on_event(
        ev(EventKind.TRANSLATION_RESULT, items=[item(1, "MỞ CỔNG MỚI", "Open the gate traveler")]),
        0.3,
    )
    assert [i.text for i in out] == ["MỞ CỔNG MỚI"]
    # The swap left no pending state behind.
    assert s.held == 1


def test_a_stale_translation_cannot_land_during_a_hold():
    """The translation requested for the wobbled text is the only one accepted."""
    s = shown_state("Open the gate, traveler.", "MỞ CỔNG")
    s.on_event(
        ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "Open the gate traveler")]), 0.2
    )
    out = s.on_event(
        ev(EventKind.TRANSLATION_RESULT, items=[item(1, "SAI", "Something else entirely")]), 0.3
    )
    assert [i.text for i in out] == ["MỞ CỔNG"]


def test_a_word_swap_is_semantic_and_drops_the_cover():
    s = shown_state("The gate is locked.", "CỔNG ĐÃ KHOÁ")
    out = s.on_event(
        ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "The gate is open.")]), 0.2
    )
    assert out == []


def test_a_digit_change_during_a_hold_still_cuts_immediately():
    """Cosmetic tolerance must never apply to numbers."""
    s = shown_state("You have 100 HP", "Bạn có 100 HP")
    out = s.on_event(
        ev(EventKind.OCR_RESULT, scan_id=2, blocks=[block(1, "You have 10 HP")]), 0.2
    )
    assert out == []
    assert s.held == 0


def test_known_rects_includes_blocks_without_a_translation():
    s = OverlayState()
    s.on_event(
        ev(
            EventKind.OCR_RESULT,
            scan_id=1,
            blocks=[block(1, "ONE"), block(2, "TWO")],
        ),
        0.0,
    )
    assert set(s.known_rects()) == {1, 2}
