import threading

import pytest

from ocrtrans.translate import (
    Envit5Engine,
    NllbEngine,
    TranslationCache,
    TranslationService,
    normalize,
)


class FakeEngine:
    name, kind = "fake", "local"

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        return [t.upper() for t in texts]


class BlockingEngine:
    name, kind = "block", "local"

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def load(self):
        pass

    def unload(self):
        pass

    def translate(self, texts):
        self.started.set()
        self.release.wait(5.0)
        return texts


def test_cache_returns_hit_flag():
    svc = TranslationService(FakeEngine(), TranslationCache())
    assert svc.translate("hello") == ("HELLO", False)
    assert svc.translate("  hello  ") == ("HELLO", True)


def test_normalize_strips_control_chars():
    assert normalize("  a\x00  b \n c ") == "a b c"


def test_nllb_rejects_missing_model_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        NllbEngine(tmp_path / "nope").load()


def test_switch_refuses_while_translate_in_flight():
    busy = BlockingEngine()
    svc = TranslationService(busy, TranslationCache())
    t = threading.Thread(target=svc.translate, args=("hello",))
    t.start()
    assert busy.started.wait(1.0)
    with pytest.raises(RuntimeError, match="engine busy"):
        svc.switch(FakeEngine())
    busy.release.set()
    t.join()


@pytest.mark.gpu
def test_nllb_translates_to_vietnamese():
    from pathlib import Path

    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download("facebook/nllb-200-distilled-600M"))
    engine = NllbEngine(model_dir)
    engine.load()
    out = engine.translate(["Open the gate."])[0]
    assert out
    assert any(ord(ch) > 127 for ch in out), out


def test_cache_evicts_least_recently_used():
    cache = TranslationCache(capacity=2)
    cache.put("a", "1")
    cache.put("b", "2")
    cache.get("a")
    cache.put("c", "3")
    assert cache.get("a") == "1"
    assert cache.get("b") is None
    assert cache.stats()["size"] == 2


class CountingEngine(FakeEngine):
    def __init__(self):
        self.calls = []

    def translate(self, texts):
        self.calls.append(tuple(texts))
        return super().translate(texts)


def test_translate_many_uses_one_engine_call_and_caches():
    engine = CountingEngine()
    service = TranslationService(engine, TranslationCache())
    first = service.translate_many(["a", "b"])
    second = service.translate_many(["a", "b"])
    assert engine.calls == [("a", "b")]
    assert first == [("A", False), ("B", False)]
    assert second == [("A", True), ("B", True)]



class LifecycleEngine:
    name, kind = "lifecycle", "local"

    def __init__(self):
        self.loads = 0
        self.unloads = 0

    def load(self):
        self.loads += 1

    def unload(self):
        self.unloads += 1

    def translate(self, texts):
        return list(texts)


def test_switch_does_not_reload_an_already_loaded_engine():
    old, new = LifecycleEngine(), LifecycleEngine()
    old.load()
    new.load()
    service = TranslationService(old, TranslationCache())
    service.switch(new)
    assert new.loads == 1
    assert old.unloads == 1
