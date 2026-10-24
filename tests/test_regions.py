from ocrtrans.config import Config
from ocrtrans.geometry import Rect
from ocrtrans.regions import (
    NormalizedRegion,
    RegionSet,
    add_region,
    clamp_rect,
    drag_rect,
    from_physical,
    handle_at,
    remap,
    remove_region,
    replace_region,
)


def cfg_with(mode: str, regions=None) -> Config:
    cfg = Config()
    cfg.region_mode = mode
    cfg.regions = regions or []
    return cfg


def test_region_roundtrip_survives_resolution_change():
    old, new = Rect(0, 0, 1920, 1080), Rect(0, 0, 2560, 1440)
    reg = from_physical(Rect(960, 810, 640, 200), old)
    rect, aspect_changed = remap(reg, old, new)
    assert rect == Rect(1280, 1080, 853, 267)
    assert aspect_changed is False


def test_single_mode_resolves_only_the_first_region():
    cfg = cfg_with("single", [{"monitor_id": "0", "x": 0.1, "y": 0.1, "w": 0.5, "h": 0.1},
                              {"monitor_id": "0", "x": 0.1, "y": 0.5, "w": 0.2, "h": 0.1}])
    assert RegionSet.from_config(cfg, Rect(0, 0, 1000, 1000)) == [Rect(100, 100, 500, 100)]


def test_multi_mode_resolves_every_region():
    cfg = cfg_with("multi", [{"monitor_id": "0", "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.1},
                             {"monitor_id": "0", "x": 0.0, "y": 0.5, "w": 0.2, "h": 0.1}])
    assert len(RegionSet.from_config(cfg, Rect(0, 0, 1000, 1000))) == 2


def test_a_region_outside_the_monitor_is_clamped():
    assert clamp_rect(Rect(900, 900, 400, 400), Rect(0, 0, 1000, 1000)) == Rect(600, 600, 400, 400)


def test_a_tiny_region_gets_the_minimum_size():
    assert clamp_rect(Rect(10, 10, 2, 2), Rect(0, 0, 1000, 1000)) == Rect(10, 10, 16, 16)


def test_region_mode_single_uses_stored_fraction():
    cfg = cfg_with("single", [{"monitor_id": "0", "x": 0.25, "y": 0.5, "w": 0.5, "h": 0.25}])
    assert RegionSet.from_config(cfg, Rect(0, 0, 1000, 800)) == [Rect(250, 400, 500, 200)]


def test_region_mode_multi_keeps_order():
    cfg = cfg_with(
        "multi",
        [
            {"monitor_id": "0", "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
            {"monitor_id": "0", "x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5},
        ],
    )
    assert RegionSet.from_config(cfg, Rect(0, 0, 1000, 800)) == [
        Rect(0, 0, 500, 400),
        Rect(500, 400, 500, 400),
    ]


def test_handle_at_finds_each_edge_and_the_body():
    rect = Rect(100, 100, 200, 100)
    assert handle_at(rect, (100, 150)) == "w"
    assert handle_at(rect, (300, 150)) == "e"
    assert handle_at(rect, (200, 100)) == "n"
    assert handle_at(rect, (200, 200)) == "s"
    assert handle_at(rect, (100, 100)) == "nw"
    assert handle_at(rect, (300, 200)) == "se"
    assert handle_at(rect, (200, 150)) == "body"
    assert handle_at(rect, (50, 50)) is None


def test_drag_body_moves_the_rect():
    assert drag_rect(Rect(100, 100, 200, 100), "body", 10, 5, Rect(0, 0, 1000, 1000)) == Rect(
        110, 105, 200, 100
    )


def test_drag_east_edge_resizes_not_moves():
    assert drag_rect(Rect(100, 100, 200, 100), "e", 40, 0, Rect(0, 0, 1000, 1000)) == Rect(
        100, 100, 240, 100
    )


def test_drag_past_the_monitor_clamps():
    assert drag_rect(Rect(100, 100, 200, 100), "w", -500, 0, Rect(0, 0, 1000, 1000)) == Rect(
        0, 100, 300, 100
    )


def test_drag_below_the_minimum_keeps_the_minimum():
    assert drag_rect(Rect(100, 100, 200, 100), "e", -500, 0, Rect(0, 0, 1000, 1000)).w == 16


def test_add_region_appends_a_normalized_rect():
    monitor = Rect(0, 0, 1000, 1000)
    out = add_region([], Rect(100, 200, 300, 100), monitor)
    assert len(out) == 1 and out[0].x == 0.1 and out[0].w == 0.3


def test_replace_region_keeps_the_order():
    monitor = Rect(0, 0, 1000, 1000)
    regions = add_region(add_region([], Rect(0, 0, 100, 50), monitor), Rect(0, 500, 100, 50), monitor)
    out = replace_region(regions, 1, Rect(0, 600, 200, 50), monitor)
    assert out[0] == regions[0] and out[1].y == 0.6 and out[1].w == 0.2


def test_remove_region_never_empties_the_list():
    monitor = Rect(0, 0, 1000, 1000)
    one = add_region([], Rect(0, 0, 100, 50), monitor)
    assert remove_region(one, 0) == one
