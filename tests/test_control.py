from ocrtrans.config import Config
from ocrtrans.ui.control import ControlWindow


def make_window(config: Config, calls: list | None = None) -> ControlWindow:
    calls = calls if calls is not None else []
    return ControlWindow(
        config,
        lambda: calls.append("start"),
        lambda: calls.append("stop"),
        lambda: calls.append("pick"),
        lambda cfg: calls.append("apply"),
        dict,
    )


def test_control_window_reflects_config(qapp):
    window = make_window(Config(engine="envit5", capture_fps=20, min_font_px=16))
    window.engine.setCurrentText("google")
    window.fps.setValue(25)
    assert window.read_config().engine == "google"
    assert window.read_config().capture_fps == 25
    window.close()


def test_start_button_calls_callback(qapp):
    calls: list = []
    window = make_window(Config(), calls)
    window.start_button.click()
    assert calls == ["start"]
    window.close()
