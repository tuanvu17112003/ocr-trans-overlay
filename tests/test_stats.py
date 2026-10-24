from ocrtrans.stats import LatencyStats


def test_percentiles_are_separated_by_cache():
    s = LatencyStats(capacity=5)  # capacity is per ring, so both fit
    for v in [10, 20, 30, 40, 50]:
        s.add(v, cached=False)
    s.add(1, cached=True)
    s.add(3, cached=True)
    assert s.percentile(50, cached=False) == 30
    assert s.percentile(50, cached=True) == 2
    assert s.snapshot()["n"] == 7
    assert s.snapshot()["cached_n"] == 2


def test_ring_evicts_oldest_per_ring():
    s = LatencyStats(capacity=3)
    for v in [1, 2, 3, 4]:
        s.add(v, cached=False)
    s.add(9, cached=True)
    assert s.snapshot()["uncached_n"] == 3
    assert s.snapshot()["cached_n"] == 1
