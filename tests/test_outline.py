from ocrtrans.geometry import Rect
from ocrtrans.ui import outline as outline_mod
from ocrtrans.ui.outline import RegionOutline


def test_outline_keeps_the_regions_and_selection(qapp):
    view = RegionOutline(Rect(0, 0, 1000, 1000))
    view.set_regions([Rect(10, 10, 100, 40), Rect(10, 60, 80, 20)], active=1, inactive=set())
    assert view.rects == [Rect(10, 10, 100, 40), Rect(10, 60, 80, 20)]
    assert view.active == 1
    view.close()


def test_outline_stays_hidden_while_running(qapp, monkeypatch):
    monkeypatch.setattr(outline_mod, "exclude_from_capture", lambda hwnd: True)
    view = RegionOutline(Rect(0, 0, 1000, 1000))
    view.set_regions([Rect(10, 10, 100, 40)], active=0, inactive=set())
    assert view.set_visible_for(stopped=False, force=False) is False
    assert view.set_visible_for(stopped=True, force=False) is True
    assert view.set_visible_for(stopped=False, force=True) is True
    assert view.set_visible_for(stopped=False, force=False) is False
    view.close()


def test_outline_hides_when_no_region_is_known(qapp, monkeypatch):
    monkeypatch.setattr(outline_mod, "exclude_from_capture", lambda hwnd: True)
    view = RegionOutline(Rect(0, 0, 1000, 1000))
    assert view.set_visible_for(stopped=True, force=True) is False
    view.close()


def test_outline_refuses_to_show_when_exclusion_fails(qapp, monkeypatch):
    monkeypatch.setattr(outline_mod, "exclude_from_capture", lambda hwnd: False)
    view = RegionOutline(Rect(0, 0, 1000, 1000))
    view.set_regions([Rect(10, 10, 100, 40)], active=0, inactive=set())
    assert view.set_visible_for(stopped=True, force=False) is False
    assert view.excluded is False
    view.close()
