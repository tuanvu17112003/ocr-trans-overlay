from ocrtrans.geometry import Rect, iou, intersects


def test_iou_identical_is_one():
    assert iou(Rect(0, 0, 10, 10), Rect(0, 0, 10, 10)) == 1.0


def test_rects_touching_at_edge_do_not_intersect():
    assert not intersects(Rect(0, 0, 10, 10), Rect(0, 10, 10, 10))
    assert not Rect(0, 0, 10, 10).intersects(Rect(0, 10, 10, 10))
