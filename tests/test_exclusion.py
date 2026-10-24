from ocrtrans.exclusion import QtHideGate


class FakeBridge:
    def __init__(self, ack):
        self.ack = ack
        self.restores = 0

    def hide_and_ack(self, timeout_ms):
        return self.ack

    def restore(self):
        self.restores += 1


def test_timeout_skips_frame_and_trips_after_three():
    g = QtHideGate(FakeBridge(ack=False))
    assert g.acquire() is False and g.tripped is False
    assert g.acquire() is False and g.acquire() is False
    assert g.tripped is True


def test_success_restores_overlay():
    b = FakeBridge(ack=True)
    g = QtHideGate(b)
    assert g.acquire() is True
    g.release()
    assert b.restores == 1
