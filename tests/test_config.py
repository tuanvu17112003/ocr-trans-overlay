import pytest

from ocrtrans.config import load_config, save_config


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
