from ocrtrans.capture import Health
from ocrtrans.geometry import Rect
from ocrtrans.lifecycle import OverlayState
from ocrtrans.pipeline import BlockView, EventKind, PipelineEvent, RenderItem


def item(block_id, text, source="X"):
    return RenderItem(block_id=block_id, text=text, rect=Rect(0, 0, 10, 10), source_text=source, cached=False)


def block(block_id, text, state="stable"):
    return BlockView(block_id=block_id, rect=Rect(0, 0, 10, 10), source_text=text, state=state)


def ev(kind, epoch=1, **kw):
    if kind is EventKind.OCR_RESULT and "observed_block_ids" not in kw:
        kw["observed_block_ids"] = [b.block_id for b in kw.get("blocks", [])] or None
    return PipelineEvent(kind=kind, epoch=epoch, at=0.0, **kw)


def test_missing_block_removed_after_two_ocr_passes():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HELLO")]), 0.0)
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[]), 0.07)
    assert s.on_event(ev(EventKind.OCR_RESULT, blocks=[]), 0.14) == []


def test_crop_scan_does_not_miss_other_blocks():
    s = OverlayState()
    s.on_event(
        ev(EventKind.OCR_RESULT, blocks=[block(1, "A"), block(2, "B")], observed_block_ids=[1, 2]),
        0.0,
    )
    for t in (0.07, 0.14):
        s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "A2")], observed_block_ids=[1]), t)
    assert s.state(2) == "shown"


def test_translation_result_never_counts_as_missing():
    s = OverlayState()
    s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HELLO")]), 0.0)
    for t in (0.1, 0.2, 0.3):
        s.on_event(ev(EventKind.TRANSLATION_RESULT, items=[item(1, "XIN CHÀO", "HELLO")]), t)
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


def test_pending_block_is_not_rendered():
    s = OverlayState()
    out = s.on_event(ev(EventKind.OCR_RESULT, blocks=[block(1, "HEL", state="pending")]), 0.0)
    assert out == []
    assert s.state(1) == "hidden"

