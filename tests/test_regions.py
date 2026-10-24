from ocrtrans.config import Config
from ocrtrans.geometry import Rect
from ocrtrans.regions import NormalizedRegion, RegionSet, clamp_rect, from_physical, remap


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
