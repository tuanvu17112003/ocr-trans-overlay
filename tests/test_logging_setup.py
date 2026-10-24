import logging

from ocrtrans.logging_setup import QuietArgos


def _record(name, level):
    return logging.LogRecord(name, level, __file__, 1, "x", None, None)


def test_argos_info_is_filtered_even_though_it_sets_its_own_level():
    quiet = QuietArgos()
    assert quiet.filter(_record("argostranslate.utils", logging.INFO)) is False


def test_argos_warnings_and_our_own_logs_pass():
    quiet = QuietArgos()
    assert quiet.filter(_record("argostranslate.utils", logging.WARNING)) is True
    assert quiet.filter(_record("ocrtrans.pipeline", logging.INFO)) is True
