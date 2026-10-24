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
