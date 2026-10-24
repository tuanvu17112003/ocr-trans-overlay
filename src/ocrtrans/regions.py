"""Region modes and normalized region storage."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .geometry import Rect


@dataclass(frozen=True)
class NormalizedRegion:
    monitor_id: str
    x: float
    y: float
    w: float
    h: float


def from_physical(rect: Rect, monitor: Rect) -> NormalizedRegion:
    return NormalizedRegion(
        monitor_id="0",
        x=rect.x / monitor.w,
        y=rect.y / monitor.h,
        w=rect.w / monitor.w,
        h=rect.h / monitor.h,
    )


def to_physical(region: NormalizedRegion, monitor: Rect) -> Rect:
    return Rect(
        monitor.x + round(region.x * monitor.w),
        monitor.y + round(region.y * monitor.h),
        round(region.w * monitor.w),
        round(region.h * monitor.h),
    )


def remap(region: NormalizedRegion, old_monitor: Rect, new_monitor: Rect) -> tuple[Rect, bool]:
    old_aspect = old_monitor.w / old_monitor.h
    new_aspect = new_monitor.w / new_monitor.h
    aspect_changed = abs(new_aspect - old_aspect) / old_aspect > 0.02
    return to_physical(region, new_monitor), aspect_changed


def clamp_rect(rect: Rect, monitor: Rect, min_size: int = 16) -> Rect:
    """Fit a capture rect inside the monitor, at least min_size on each side."""
    w = min(max(rect.w, min_size), monitor.w)
    h = min(max(rect.h, min_size), monitor.h)
    x = min(max(rect.x, monitor.x), monitor.right - w)
    y = min(max(rect.y, monitor.y), monitor.bottom - h)
    return Rect(x, y, w, h)


class RegionSet:
    @staticmethod
    def from_config(cfg: Config, monitor: Rect) -> list[Rect]:
        if not cfg.regions:
            return [Rect(monitor.x, monitor.y, monitor.w, monitor.h)]
        regions = [NormalizedRegion(**r) for r in cfg.regions]
        if cfg.region_mode == "single":
            regions = regions[:1]
        return [clamp_rect(to_physical(r, monitor), monitor) for r in regions]
