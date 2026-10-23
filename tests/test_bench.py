from ocrtrans.bench.perf import _percentiles
from ocrtrans.bench.quality import score_quality


def test_number_preserved():
    assert score_quality([("3 HP", "3 HP")])["number_preserved"] == 1.0
    assert score_quality([("3 HP", "ba HP")])["number_preserved"] == 0.0


def test_report_shape():
    report = score_quality([("Open the gate", "Mở cổng")])
    assert set(report) == {
        "chrf_pp",
        "comet",
        "number_preserved",
        "proper_noun_preserved",
        "length_ratio",
        "empty_count",
    }


def test_percentiles_shape():
    result = _percentiles([1.0, 2.0, 3.0, 4.0])
    assert set(result) == {"p50", "p95", "p99"}
    assert result["p50"] <= result["p95"] <= result["p99"]
