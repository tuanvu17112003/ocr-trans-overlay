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


def handle_at(rect: Rect, point: tuple[int, int], tolerance: int = 6) -> str | None:
    """Name the handle under a screen point: an edge, a corner, body or None."""
    x, y = point
    left, right = rect.x, rect.right
    top, bottom = rect.y, rect.bottom
    if not (
        left - tolerance <= x <= right + tolerance and top - tolerance <= y <= bottom + tolerance
    ):
        return None
    near_l, near_r = abs(x - left) <= tolerance, abs(x - right) <= tolerance
    near_t, near_b = abs(y - top) <= tolerance, abs(y - bottom) <= tolerance
    if near_t and near_l:
        return "nw"
    if near_t and near_r:
        return "ne"
    if near_b and near_l:
        return "sw"
    if near_b and near_r:
        return "se"
    if near_l:
        return "w"
    if near_r:
        return "e"
    if near_t:
        return "n"
    if near_b:
        return "s"
    if left <= x <= right and top <= y <= bottom:
        return "body"
    return None


def drag_rect(
    rect: Rect, handle: str, dx: int, dy: int, monitor: Rect, min_size: int = 16
) -> Rect:
    """Move or resize a rect by a drag delta, staying inside the monitor."""
    if handle == "body":
        return clamp_rect(Rect(rect.x + dx, rect.y + dy, rect.w, rect.h), monitor, min_size)
    left, top, right, bottom = rect.x, rect.y, rect.right, rect.bottom
    if "w" in handle:
        left = min(max(monitor.x, left + dx), right - min_size)
    if "e" in handle:
        right = max(min(monitor.right, right + dx), left + min_size)
    if "n" in handle:
        top = min(max(monitor.y, top + dy), bottom - min_size)
    if "s" in handle:
        bottom = max(min(monitor.bottom, bottom + dy), top + min_size)
    return Rect(left, top, right - left, bottom - top)


def add_region(
    regions: list[NormalizedRegion], rect: Rect, monitor: Rect, min_size: int = 16
) -> list[NormalizedRegion]:
    clamped = clamp_rect(rect, monitor, min_size)
    return [*regions, from_physical(clamped, monitor)]


def replace_region(
    regions: list[NormalizedRegion], index: int, rect: Rect, monitor: Rect, min_size: int = 16
) -> list[NormalizedRegion]:
    out = list(regions)
    out[index] = from_physical(clamp_rect(rect, monitor, min_size), monitor)
    return out


def remove_region(regions: list[NormalizedRegion], index: int) -> list[NormalizedRegion]:
    if len(regions) <= 1:
        return list(regions)
    return [region for position, region in enumerate(regions) if position != index]


class RegionSet:
    @staticmethod
    def from_config(cfg: Config, monitor: Rect) -> list[Rect]:
        if not cfg.regions:
            return [Rect(monitor.x, monitor.y, monitor.w, monitor.h)]
        regions = [NormalizedRegion(**r) for r in cfg.regions]
        if cfg.region_mode == "single":
            regions = regions[:1]
        return [clamp_rect(to_physical(r, monitor), monitor) for r in regions]
