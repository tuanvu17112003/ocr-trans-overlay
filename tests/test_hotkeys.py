import pytest

from ocrtrans.hotkeys import MOD_ALT, MOD_CONTROL, parse_binding


def test_parse_binding_modifiers_and_key():
    assert parse_binding("Ctrl+Alt+Z") == (MOD_CONTROL | MOD_ALT, ord("Z"))


def test_invalid_binding_raises():
    with pytest.raises(ValueError):
        parse_binding("Ctrl+Nope")
