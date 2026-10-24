import queue

from ocrtrans.config import Config
from ocrtrans.geometry import Rect
from ocrtrans.overlay import OverlayWindow
from ocrtrans.pipeline import RenderItem


def make_item(text="Xin chào", rect=Rect(100, 100, 200, 80)):
    return RenderItem(
        block_id=1,
        text=text,
        rect=rect,
        source_text="Hello world",
        cached=False,
        line_px=20,
    )


def make_overlay(qapp):
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), queue.Queue())
    overlay.show_items([make_item()])
    return overlay


def test_overlay_picks_up_a_font_change(qapp):
    cfg = Config()
    view = OverlayWindow(Rect(0, 0, 800, 200), cfg, queue.Queue())
    view.show_items([make_item()])
    cfg.font_family = "Consolas"
    view.show_items([make_item()])
    assert view.measure.family == "Consolas"
    view.close()


def test_cover_is_the_source_block(qapp):
    overlay = make_overlay(qapp)
    layout = overlay._layouts[0]
    assert layout.cover == Rect(100, 100, 200, 80)
    assert layout.rect.x >= 100 and layout.rect.y >= 100
    assert layout.rect.x + layout.rect.w <= 300
    assert layout.rect.y + layout.rect.h <= 180
    overlay.close()


def test_text_is_centered_vertically_inside_the_cover(qapp):
    overlay = make_overlay(qapp)
    layout = overlay._layouts[0]
    top_gap = layout.rect.y - 100
    bottom_gap = 180 - (layout.rect.y + layout.rect.h)
    assert abs(top_gap - bottom_gap) <= 2
    overlay.close()


def test_background_fills_the_cover_with_opaque_color(qapp):
    overlay = make_overlay(qapp)
    image = overlay.grab().toImage()
    expected = Config().bg_color.lower()
    for x, y in ((102, 102), (297, 102), (102, 177), (297, 177)):
        color = image.pixelColor(x, y)
        assert color.alpha() == 255, (x, y)
        assert color.name() == expected, (x, y, color.name())
    overlay.close()


def test_an_unchanged_snapshot_does_not_repaint(qapp):
    from ocrtrans.pipeline import BlockView, EventKind, PipelineEvent

    events = queue.Queue()
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), events)
    view = BlockView(1, Rect(100, 100, 200, 80), "Hello world", "stable", 20)
    events.put(PipelineEvent(kind=EventKind.OCR_RESULT, epoch=1, at=0.0, scan_id=1, blocks=[view]))
    events.put(
        PipelineEvent(
            kind=EventKind.TRANSLATION_RESULT,
            epoch=1,
            at=0.0,
            scan_id=1,
            items=[make_item(text="Xin chao")],
        )
    )
    calls = []
    original = overlay._recompute

    def counted():
        calls.append(1)
        original()

    overlay._recompute = counted
    overlay._drain()
    assert calls == [1]

    events.put(PipelineEvent(kind=EventKind.OCR_RESULT, epoch=1, at=0.0, scan_id=2, blocks=[view]))
    overlay._drain()
    assert calls == [1], "an identical snapshot repainted the whole overlay"
    overlay.close()


def test_the_overlay_window_shrinks_to_the_selected_region(qapp):
    overlay = OverlayWindow(Rect(0, 0, 2560, 1440), Config(), queue.Queue())
    overlay.set_regions([Rect(63, 1245, 1760, 53)])
    assert overlay.screen_rect == Rect(55, 1237, 1776, 69 + overlay.GROW_PX)
    overlay.close()


def test_damage_is_the_block_area_not_the_whole_screen(qapp):
    overlay = OverlayWindow(Rect(0, 0, 2560, 1440), Config(), queue.Queue())
    item = make_item(rect=Rect(100, 200, 300, 60))
    damage = overlay._damage([], [item])
    assert damage.width() <= 304
    assert damage.height() <= item.rect.h + overlay.GROW_PX + 4
    assert damage.width() < 2560
    overlay.close()


def test_a_grown_cover_covers_the_extra_lines(qapp):
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), queue.Queue())
    tall = make_item(text="một hai ba bốn năm sáu bảy tám chín mười", rect=Rect(100, 100, 200, 24))
    overlay.show_items([tall])
    layout = overlay._layouts[0]
    assert layout.cover.h > 24
    assert layout.cover.bottom <= overlay.screen_rect.bottom
    overlay.close()


def test_a_block_below_caps_the_growth(qapp):
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), queue.Queue())
    overlay.set_regions([Rect(80, 80, 240, 300)])
    above = make_item(text="một hai ba bốn năm sáu bảy tám chín mười", rect=Rect(100, 100, 200, 24))
    below = RenderItem(
        block_id=2, text="b", rect=Rect(100, 146, 200, 20), source_text="b", cached=False, line_px=20
    )
    overlay.show_items([above, below])
    assert overlay._layouts[0].cover.h > 24
    assert overlay._layouts[0].cover.bottom <= 142
    overlay.close()


def test_damage_covers_the_grown_box(qapp):
    overlay = OverlayWindow(Rect(0, 0, 800, 600), Config(), queue.Queue())
    item = make_item(text="một hai ba bốn năm sáu bảy tám chín mười", rect=Rect(100, 100, 200, 24))
    damage = overlay._damage([], [item])
    assert damage.height() >= item.rect.h + overlay.GROW_PX
    overlay.close()
