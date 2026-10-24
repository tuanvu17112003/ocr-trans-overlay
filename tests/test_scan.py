import numpy as np
import pytest

from ocrtrans.scan import ScanBudget, frame_changed


@pytest.mark.parametrize(
    "fps,last_ms,empty,wake,expected",
    [
        (15, 300, 0, False, 0.450),
        (15, 300, 0, True, 0.450),
        (15, 300, 3, False, 1.000),
        (15, 300, 3, True, 0.450),
        (15, 852, 3, False, 1.278),
        (15, 852, 3, True, 1.278),
        (15, 20, 0, False, 0.250),
        (15, 20, 5, False, 1.000),
    ],
)
def test_scan_budget_matches_the_pinned_table(fps, last_ms, empty, wake, expected):
    assert ScanBudget(fps).interval(last_ms, empty, wake) == pytest.approx(expected)


def test_ttl_is_two_intervals_clamped_between_400ms_and_1s():
    b = ScanBudget(15)
    assert b.ttl(0.25) == pytest.approx(0.5)
    assert b.ttl(0.1) == pytest.approx(0.4)
    assert b.ttl(0.852) == pytest.approx(1.0)


def test_set_fps_changes_the_floor():
    b = ScanBudget(15)
    b.set_fps(5)
    assert b.interval(20, 0, False) == pytest.approx(0.25)
    b.set_fps(30)
    assert b.interval(20, 0, False) == pytest.approx(0.25)


def test_frame_changed_flags_noise_but_not_a_repeat():
    frame = np.full((100, 320, 3), 30, np.uint8)
    changed, gray = frame_changed(None, frame)
    assert changed is False
    assert gray.shape[0] == 100
    assert frame_changed(gray, frame)[0] is False
    noise = np.random.default_rng(0).integers(0, 255, frame.shape, dtype=np.uint8)
    assert frame_changed(gray, noise)[0] is True


def test_frame_changed_survives_a_region_size_change():
    _, gray = frame_changed(None, np.full((100, 320, 3), 30, np.uint8))
    changed, new_gray = frame_changed(gray, np.full((100, 400, 3), 30, np.uint8))
    assert changed is False
    assert new_gray.shape != gray.shape
