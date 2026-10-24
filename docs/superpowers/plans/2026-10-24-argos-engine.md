# Optional Argos en->vi engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a free, offline, CPU-only English->Vietnamese engine (`argostranslate`) that the app can use when the GPU models do not fit, without making it a required dependency.

**Architecture:** A thin `ArgosEngine` behind the existing `Engine` protocol. The package is installed on first `load()` from Argos's package index and cached in the translation object. The app, config and engine combo pick it up through the existing `ENGINES` set; nothing else changes.

**Tech Stack:** Python 3.13, optional `argostranslate` (~100 MB model, CPU), pytest.

**Spec:** `docs/superpowers/specs/2026-10-24-text-first-overlay-design.md` section 7.

## Global Constraints

- Offline after the first model download; no API key, no account.
- No new required dependency: importing the app must never import `argostranslate`.
- The engine must be removable (uninstall the package) without breaking the app; selecting `argos` without the package shows a clear error.
- Code/comments/commits in English.

## Review Focus

1. **Package not installed** -> error names the install command, app stays alive (Task 1: `test_missing_dependency_raises_a_clear_error`).
2. **Model not downloaded yet** -> first `load()` installs `en->vi`, second `load()` does not reinstall (Task 1: `test_load_installs_the_model_once`).
3. **Batch order** -> translations come back in input order (Task 1: `test_translate_preserves_order`).
4. **Engine selection with the package missing** -> falls back to the existing error path in `load_engine`, no crash (Task 2: config accepts `argos`; `build_engine` imports lazily).

---

### Task 1: ArgosEngine

**Files:**
- Create: `src/ocrtrans/argos.py`
- Test: `tests/test_argos.py`

**Interfaces:**
- Consumes: `translate.Engine` protocol shape (`name`, `kind`, `load`, `unload`, `translate(texts) -> list[str]`).
- Produces: `argos.ArgosEngine(from_code="en", to_code="vi")` with `name = "argos"`, `kind = "local"`, `load()`, `unload()`, `translate(texts: list[str]) -> list[str]`, and `ArgosEngine.installed_translation` (the cached object, `None` before load).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_argos.py
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
    with pytest.raises(RuntimeError, match="pip install argostranslate"):
        ArgosEngine().load()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_argos.py -q`
Expected: `ModuleNotFoundError: ocrtrans.argos`.

- [ ] **Step 3: Implement `src/ocrtrans/argos.py`**

`load()` imports inside the method, raises `RuntimeError("argostranslate is not installed; run: pip install argostranslate")` on `ImportError`, and is idempotent (return early when the translation object is already cached). Resolve the translation object through a helper that wraps `translate.get_translation_from_codes(from_code, to_code)` in `try/except (AttributeError, StopIteration)` and returns `None` on failure — the real API **raises** `AttributeError` when no language packs are installed at all, it does not return `None`. When `None`: call `update_package_index()`, pick `from_code=="en" and to_code=="vi"` from `get_available_packages()`, call `install()`, and re-read through the same helper; wrap index/install errors in `RuntimeError("could not install the Argos en->vi model: ...")` (the real `get_available_packages` can recurse when the index is missing and the download fails). If the re-read is still `None`, raise `RuntimeError("Argos en->vi model is not available")`. `translate` maps the cached object over the list. `unload()` drops the reference.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_argos.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/argos.py tests/test_argos.py
git commit -m "feat: optional offline Argos en-vi engine"
```

---

### Task 2: Config, engine selection, README

**Files:**
- Modify: `src/ocrtrans/config.py:10` (`ENGINES`)
- Modify: `src/ocrtrans/app.py:58-75` (`build_engine`)
- Modify: `README.md` (engine tables)
- Modify: `tests/test_config.py` (append; do not replace the existing roundtrip/rejection tests)

**Interfaces:**
- Consumes: `argos.ArgosEngine` from Task 1.
- Produces: `"argos"` accepted by `Config`; `build_engine` returns a loaded `ArgosEngine` for `engine="argos"`.

- [ ] **Step 1: Write the failing tests**

Append to the existing `tests/test_config.py`:

```python
from ocrtrans.config import Config, _validate


def test_argos_is_a_valid_engine():
    _validate(Config(engine="argos"))


def test_build_engine_returns_argos_before_the_consent_gate(monkeypatch):
    from ocrtrans import app as app_module

    class FakeArgos:
        def __init__(self):
            self.loaded = False

        def load(self):
            self.loaded = True

    monkeypatch.setattr("ocrtrans.argos.ArgosEngine", FakeArgos)
    engine = app_module.build_engine(Config(engine="argos", online_consent=False))
    assert isinstance(engine, FakeArgos) and engine.loaded
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -q`
Expected: `ValueError: engine: 'argos' ...`, then `build_engine` falls back to NLLB (or `AttributeError` before the config change).

- [ ] **Step 3: Implement**

Add `"argos"` to `ENGINES`. In `build_engine`, insert the branch **immediately after the `if cfg.engine in LOCAL_REPOS:` block and before the `if not cfg.online_consent:` guard** (otherwise the consent guard silently returns NLLB):

```python
if cfg.engine == "argos":
    from .argos import ArgosEngine

    engine = ArgosEngine()
    engine.load()
    return engine
```

`ui/control.py` already fills the combo from `sorted(ENGINES)`, so no UI change. In the README engine table add `| Argos en->vi | local (CPU) | unlimited | ~100 MB, downloads once; lower quality than NLLB |`, and in the install section mark `argostranslate` as optional.

- [ ] **Step 4: Run the suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocrtrans/config.py src/ocrtrans/app.py README.md tests/test_config.py
git commit -m "feat: select the offline Argos engine in config and the app"
```
