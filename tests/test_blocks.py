from ocrtrans.blocks import BlockMemory, BlockState, match_score
from ocrtrans.geometry import Box, Rect


def box(text, x=0, y=0, w=100, h=20, line_px=18):
    return Box(Rect(x, y, w, h), text, 0.9, line_px=line_px)


def test_same_text_becomes_stable_after_the_safe_window():
    m = BlockMemory("safe")
    assert m.observe([box("HELLO")], 0.0, 0.4)[0].state is BlockState.PENDING
    assert m.observe([box("HELLO")], 0.20, 0.4)[0].state is BlockState.PENDING
    assert m.observe([box("HELLO")], 0.31, 0.4)[0].state is BlockState.STABLE


def test_fast_profile_uses_a_shorter_window():
    m = BlockMemory("fast")
    m.observe([box("HELLO")], 0.0, 0.4)
    assert m.observe([box("HELLO")], 0.16, 0.4)[0].state is BlockState.STABLE


def test_one_digit_change_resets_stability():
    m = BlockMemory("fast")
    m.observe([box("You have 100 HP")], 0.0, 1.0)
    stable = m.observe([box("You have 100 HP")], 0.2, 1.0)[0]
    assert stable.state is BlockState.STABLE
    changed = m.observe([box("You have 10 HP")], 0.3, 1.0)[0]
    assert changed.text == "You have 10 HP"
    assert changed.state is BlockState.PENDING
    assert changed.first_seen == 0.3
    assert m.observe([box("You have 10 HP")], 0.4, 1.0)[0].state is BlockState.PENDING


def test_case_and_spacing_wobble_is_the_same_text():
    m = BlockMemory("fast")
    m.observe([box("HELLO WORLD")], 0.0, 1.0)
    block = m.observe([box("  hello   world ")], 0.2, 1.0)[0]
    assert block.state is BlockState.STABLE


def test_identity_survives_small_rect_drift():
    m = BlockMemory("fast")
    first = m.observe([box("HELLO", x=10, y=10)], 0.0, 1.0)[0]
    second = m.observe([box("HELLO", x=12, y=11)], 0.1, 1.0)[0]
    assert first.id == second.id


def test_two_boxes_never_match_the_same_block():
    m = BlockMemory("fast")
    m.observe([box("A", y=10), box("B", y=60)], 0.0, 1.0)
    blocks = m.observe([box("B", y=10), box("A", y=60)], 0.1, 1.0)
    by_rect = {b.rect.y: b.text for b in blocks}
    assert by_rect == {10: "B", 60: "A"}
    assert len({b.id for b in blocks}) == 2


def test_block_survives_one_empty_pass_within_ttl():
    m = BlockMemory("fast")
    m.observe([box("HELLO")], 0.0, 0.5)
    assert [b.text for b in m.observe([], 0.3, 0.5)] == ["HELLO"]
    assert m.observe([], 0.51, 0.5) == []


def test_a_moved_block_with_new_text_does_not_ghost_at_the_old_rect():
    m = BlockMemory("fast")
    m.observe([box("AAAA", x=0, w=100)], 0.0, 1.0)
    blocks = m.observe([box("BBBB", x=60, w=100)], 0.1, 1.0)
    assert [(b.rect.x, b.text) for b in blocks] == [(60, "BBBB")]


def test_match_score_prefers_overlap_and_containment():
    a = Rect(0, 0, 100, 20)
    assert match_score(a, Rect(0, 0, 100, 20)) == 1.0
    assert match_score(a, Rect(0, 0, 100, 24)) >= 0.6
    assert match_score(a, Rect(500, 500, 10, 10)) == 0.0
