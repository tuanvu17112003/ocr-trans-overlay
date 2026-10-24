import logging

from ocrtrans.logging_setup import QuietArgos, set_debug


def _record(name, level):
    return logging.LogRecord(name, level, __file__, 1, "x", None, None)


def test_debug_toggles_the_root_level_and_keeps_argos_quiet():
    root = logging.getLogger()
    before = root.level
    try:
        set_debug(True)
        assert root.level == logging.DEBUG
        set_debug(False)
        assert root.level == logging.INFO
        assert logging.getLogger("argostranslate").level == logging.WARNING
    finally:
        root.setLevel(before)


def test_argos_info_is_filtered_even_though_it_sets_its_own_level():
    quiet = QuietArgos()
    assert quiet.filter(_record("argostranslate.utils", logging.INFO)) is False


def test_argos_warnings_and_our_own_logs_pass():
    quiet = QuietArgos()
    assert quiet.filter(_record("argostranslate.utils", logging.WARNING)) is True
    assert quiet.filter(_record("ocrtrans.pipeline", logging.INFO)) is True
