from ocrtrans.fit import fit_text
from ocrtrans.geometry import Rect


class MonoMeasure:
    def measure(self, text, font_px):
        return (len(text) * font_px // 2, font_px + 4)

    def line_height(self, font_px):
        return font_px + 4


def test_grows_before_shrinking_or_truncating():
    r = fit_text("một hai ba bốn năm sáu", Rect(0, 500, 200, 40), Rect(0, 0, 1920, 1080), MonoMeasure())
    assert r.font_px == 32 and r.truncated is False and r.rect.h >= 40


def test_respects_font_floor_then_truncates():
    r = fit_text("x" * 400, Rect(0, 0, 100, 20), Rect(0, 0, 200, 100), MonoMeasure(), min_font_px=12)
    assert r.font_px == 12 and r.truncated is True


def test_never_overlaps_obstacle():
    r = fit_text(
        "một hai ba",
        Rect(0, 0, 200, 20),
        Rect(0, 0, 1920, 1080),
        MonoMeasure(),
        obstacles=[Rect(0, 24, 200, 200)],
    )
    assert r.rect.bottom <= 24
