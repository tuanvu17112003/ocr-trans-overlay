import sys
import types

import pytest

from ocrtrans.argos import ArgosEngine


def fake_argos(monkeypatch):
    installed = []
    packages = []
    calls = {"installs": 0, "translate": []}

    class Package:
        from_code, to_code, version = "en", "vi", "1.9"

        def install(self):
            calls["installs"] += 1
            installed.append(self)

    class Translation:
        def translate(self, text):
            calls["translate"].append(text)
            return f"vi:{text}"

    translation = Translation()
    package_mod = types.ModuleType("argostranslate.package")
    package_mod.get_available_packages = lambda: list(packages)
    package_mod.update_package_index = lambda: None
    translation_mod = types.ModuleType("argostranslate.translate")

    def get_translation_from_codes(f, t):
        if not installed:
            # the real API raises when no language packs are installed at all
            raise AttributeError("'NoneType' object has no attribute 'get_translation'")
        return translation

    translation_mod.get_translation_from_codes = get_translation_from_codes
    root = types.ModuleType("argostranslate")
    root.package = package_mod
    root.translate = translation_mod
    monkeypatch.setitem(sys.modules, "argostranslate", root)
    monkeypatch.setitem(sys.modules, "argostranslate.package", package_mod)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", translation_mod)
    packages.append(Package())
    return calls


def test_load_installs_the_model_once(monkeypatch):
    calls = fake_argos(monkeypatch)
    engine = ArgosEngine()
    engine.load()
    engine.load()
    assert calls["installs"] == 1
    assert engine.installed_translation is not None


def test_translate_preserves_order(monkeypatch):
    fake_argos(monkeypatch)
    engine = ArgosEngine()
    engine.load()
    assert engine.translate(["one", "two"]) == ["vi:one", "vi:two"]


def test_missing_dependency_raises_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "argostranslate", None)
    with pytest.raises(RuntimeError, match="pip install --no-deps argostranslate"):
        ArgosEngine().load()
