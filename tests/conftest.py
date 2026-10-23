import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def pytest_addoption(parser):
    parser.addoption(
        "--run-real",
        action="store_true",
        default=False,
        help="run tests that need the game window open",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "real: needs the running game window")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-real"):
        return
    skip = pytest.mark.skip(reason="needs --run-real and the game open in borderless mode")
    for item in items:
        if item.get_closest_marker("real"):
            item.add_marker(skip)
