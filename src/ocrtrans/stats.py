"""Latency samples with separate rings for cached and uncached translations."""

from __future__ import annotations

import math
from collections import deque


class LatencyStats:
    def __init__(self, capacity: int = 200) -> None:
        self._capacity = capacity
        self._cached: deque[float] = deque(maxlen=capacity)
        self._uncached: deque[float] = deque(maxlen=capacity)

    def add(self, ms: float, cached: bool) -> None:
        (self._cached if cached else self._uncached).append(float(ms))

    def _series(self, cached: bool | None) -> list[float]:
        if cached is None:
            return list(self._uncached) + list(self._cached)
        return list(self._cached if cached else self._uncached)

    def percentile(self, q: float, cached: bool | None = None) -> float:
        values = sorted(self._series(cached))
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * (q / 100.0)
        low, high = math.floor(pos), math.ceil(pos)
        if low == high:
            return values[low]
        return values[low] + (values[high] - values[low]) * (pos - low)

    def snapshot(self) -> dict:
        return {
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "n": len(self._cached) + len(self._uncached),
            "cached_n": len(self._cached),
            "uncached_n": len(self._uncached),
        }

    def reset(self) -> None:
        self._cached.clear()
        self._uncached.clear()
