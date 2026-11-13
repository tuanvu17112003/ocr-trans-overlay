"""Small counter/histogram registry behind the stats line and the HUD.

The old `stats()` dict mixed pipeline, capture and overlay values, was rebuilt
from three sources on every read, and could not answer "why" questions: how many
passes were skipped and for what reason, how many blocks the gate rejected, how
often the cover was held instead of torn down. Those numbers are what makes the
next regression visible instead of reported as "it feels flickery".
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


class Counter:
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0

    def add(self, amount: int = 1) -> None:
        self.value += amount


@dataclass
class Histogram:
    """Fixed-capacity sample ring with percentiles."""

    capacity: int = 200
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def __post_init__(self) -> None:
        if self.samples.maxlen != self.capacity:
            self.samples = deque(self.samples, maxlen=self.capacity)

    def add(self, value: float) -> None:
        self.samples.append(float(value))

    def percentile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q / 100.0
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    def snapshot(self) -> dict:
        return {
            "n": len(self.samples),
            "p50": round(self.percentile(50), 2),
            "p95": round(self.percentile(95), 2),
        }


class Metrics:
    """Thread-safe registry, cheap enough to touch on every pass."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._started = time.monotonic()

    def counter(self, name: str, amount: int = 1) -> None:
        with self._lock:
            bucket = self._counters.get(name)
            if bucket is None:
                bucket = self._counters[name] = Counter()
            bucket.add(amount)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            bucket = self._histograms.get(name)
            if bucket is None:
                bucket = self._histograms[name] = Histogram()
            bucket.add(value)

    def get(self, name: str) -> int:
        with self._lock:
            bucket = self._counters.get(name)
            return 0 if bucket is None else bucket.value

    def histogram(self, name: str) -> Histogram:
        with self._lock:
            bucket = self._histograms.get(name)
            if bucket is None:
                bucket = self._histograms[name] = Histogram()
            return bucket

    def snapshot(self) -> dict:
        with self._lock:
            counters = {name: bucket.value for name, bucket in self._counters.items()}
            histograms = {
                name: bucket.snapshot() for name, bucket in self._histograms.items()
            }
        return {
            "uptime_s": round(time.monotonic() - self._started, 1),
            "counters": counters,
            "histograms": histograms,
        }

    def reset(self) -> None:
        with self._lock:
            for bucket in self._counters.values():
                bucket.value = 0
            for bucket in self._histograms.values():
                bucket.samples.clear()


class MetricsLog:
    """Writes one JSON line per interval so a bug report carries real numbers."""

    def __init__(self, path: Path, interval_s: float = 5.0) -> None:
        self.path = Path(path)
        self.interval_s = interval_s
        self._last = 0.0

    def maybe_write(self, metrics: Metrics, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last < self.interval_s:
            return False
        self._last = now
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics.snapshot(), ensure_ascii=False) + "\n")
        except OSError:
            return False
        return True
