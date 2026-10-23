from ocrtrans.fit import fit_text
from ocrtrans.geometry import Rect


class MonoMeasure:
    def measure(self, text, font_px):
        return (len(text) * font_px // 2, font_px + 4)

    def line_height(self, font_px):
        return font_px + 4


def test_shrinks_instead_of_growing():
    # replaces test_grows_before_shrinking_or_truncating from the old suite
    r = fit_text("một hai ba bốn năm sáu", Rect(0, 500, 200, 40), Rect(0, 0, 1920, 1080), MonoMeasure())
    assert r.font_px < 32 and r.rect.h <= 40


def test_font_follows_line_height_not_block_height():
    one_line = fit_text("Hello", Rect(0, 0, 300, 20), Rect(0, 0, 1000, 800), MonoMeasure(), line_px=20)
    paragraph = fit_text("Hello", Rect(0, 0, 300, 100), Rect(0, 0, 1000, 800), MonoMeasure(), line_px=20)
    assert one_line.font_px == paragraph.font_px


def test_fit_never_leaves_the_source_box():
    result = fit_text(
        "a very long translated line that must wrap many times",
        Rect(50, 50, 200, 24),
        Rect(0, 0, 1000, 800),
        MonoMeasure(),
        line_px=24,
    )
    assert result.rect.x == 50 and result.rect.w <= 200
    assert result.rect.h <= 24
    assert result.truncated


def test_respects_font_floor_then_truncates():
    r = fit_text("x" * 400, Rect(0, 0, 100, 20), Rect(0, 0, 200, 100), MonoMeasure(), min_font_px=12)
    assert r.font_px == 12 and r.truncated is True

