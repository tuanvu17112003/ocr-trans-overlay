import pytest

from ocrtrans.config import Config, load_config, save_config


def test_defaults_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    cfg = load_config(p)
    assert cfg.capture_fps == 15 and cfg.engine == "nllb"
    cfg.engine = "gemini"
    save_config(cfg, p)
    assert load_config(p).engine == "gemini"


def test_unknown_engine_rejected(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"engine": "bogus"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_old_monitor_config_loads_as_single(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"region_mode": "monitor", "regions": [{"monitor_id": "0", "x": 0.0, "y": 0.1, "w": 0.5, "h": 0.1}]}',
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.region_mode == "single"
    assert len(cfg.regions) == 1


def test_monitor_mode_is_no_longer_writable(tmp_path):
    cfg = Config(region_mode="monitor")
    with pytest.raises(ValueError):
        save_config(cfg, tmp_path / "config.json")


def test_argos_is_a_valid_engine():
    from ocrtrans.config import Config, _validate

    _validate(Config(engine="argos"))


def test_build_engine_returns_argos_before_the_consent_gate(monkeypatch):
    from ocrtrans import app as app_module
    from ocrtrans.config import Config

    class FakeArgos:
        def __init__(self):
            self.loaded = False

        def load(self):
            self.loaded = True

    monkeypatch.setattr("ocrtrans.argos.ArgosEngine", FakeArgos)
    engine = app_module.build_engine(Config(engine="argos", online_consent=False))
    assert isinstance(engine, FakeArgos) and engine.loaded


def test_nllb_falls_back_to_argos_without_transformers(monkeypatch):
    from ocrtrans import app as app_module
    from ocrtrans.config import Config

    loaded = []

    class FakeArgos:
        name = "argos"

        def load(self):
            loaded.append(True)

    def boom(name):
        raise ImportError("no transformers")

    monkeypatch.setattr(app_module, "build_local", boom)
    monkeypatch.setattr("ocrtrans.argos.ArgosEngine", FakeArgos)
    engine = app_module.build_engine(Config(engine="nllb"))
    assert engine.name == "argos"
    assert loaded == [True]


def test_nllb_is_used_when_transformers_is_present(monkeypatch):
    from ocrtrans import app as app_module
    from ocrtrans.config import Config

    class FakeNllb:
        name = "nllb"

    monkeypatch.setattr(app_module, "build_local", lambda name: FakeNllb())
    engine = app_module.build_engine(Config(engine="nllb"))
    assert engine.name == "nllb"
