"""Metrics registry: counters, histograms, and the JSONL writer."""

import json

from ocrtrans.metrics import Histogram, Metrics, MetricsLog


def test_counters_accumulate_and_default_to_zero():
    m = Metrics()
    assert m.get("missing") == 0
    m.counter("scan_passes")
    m.counter("scan_passes")
    m.counter("scan_passes", 3)
    assert m.get("scan_passes") == 5


def test_histogram_percentiles():
    h = Histogram(capacity=10)
    for value in range(1, 11):
        h.add(value)
    assert h.percentile(50) == 5.5
    assert h.percentile(0) == 1
    assert h.percentile(100) == 10
    assert h.snapshot()["n"] == 10


def test_histogram_is_bounded():
    h = Histogram(capacity=3)
    for value in range(100):
        h.add(value)
    assert h.snapshot()["n"] == 3
    assert h.percentile(0) == 97


def test_snapshot_shape():
    m = Metrics()
    m.counter("a", 2)
    m.observe("latency", 12.5)
    snap = m.snapshot()
    assert snap["counters"]["a"] == 2
    assert snap["histograms"]["latency"]["n"] == 1
    assert "uptime_s" in snap


def test_reset_clears_everything():
    m = Metrics()
    m.counter("a")
    m.observe("h", 1.0)
    m.reset()
    assert m.get("a") == 0
    assert m.snapshot()["histograms"]["h"]["n"] == 0


def test_metrics_log_respects_the_interval(tmp_path):
    m = Metrics()
    m.counter("x")
    log = MetricsLog(tmp_path / "metrics.jsonl", interval_s=60.0)
    assert log.maybe_write(m, force=True) is True
    assert log.maybe_write(m) is False, "the writer ignored its interval"
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["counters"]["x"] == 1


def test_metrics_is_thread_safe_enough_to_hammer():
    import threading

    m = Metrics()

    def bump() -> None:
        for _ in range(500):
            m.counter("hits")
            m.observe("latency", 1.0)

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert m.get("hits") == 2000
    assert m.snapshot()["histograms"]["latency"]["n"] == 200
