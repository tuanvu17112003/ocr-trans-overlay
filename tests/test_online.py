from ocrtrans.online import GoogleEngine, OnlineRouter


class Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


class StubSession:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return Resp(self.status, self.payload)

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        return Resp(self.status, self.payload)


class FakeEngine:
    name, kind = "fake", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return [t.upper() for t in texts]


def test_google_parses_nested_array():
    body = [[["Xin chào", "Hello", None, None]], None, "en"]
    assert GoogleEngine(StubSession(payload=body)).translate(["Hello"]) == ["Xin chào"]


def test_429_marks_provider_degraded_for_cooldown():
    router = OnlineRouter(GoogleEngine(StubSession(status=429)), FakeEngine())
    out, degraded = router.translate("Hello")
    assert degraded is True and out == "HELLO"
    assert router.healthy is False

from ocrtrans.online import OnlineRouter
from ocrtrans.translate import TranslationCache


class LifecycleEngine:
    name, kind = "lifecycle", "local"

    def __init__(self):
        self.unloads = 0

    def load(self):
        pass

    def unload(self):
        self.unloads += 1

    def translate(self, texts):
        return list(texts)


def test_router_unload_frees_the_local_fallback():
    primary, fallback = LifecycleEngine(), LifecycleEngine()
    router = OnlineRouter(primary, fallback)
    router.unload()
    assert primary.unloads == 1
    assert fallback.unloads == 1
