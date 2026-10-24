from ocrtrans.config import Config
from ocrtrans.geometry import Rect
from ocrtrans.ui.control import ControlWindow

ONE = {"monitor_id": "0", "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.1}
TWO = {"monitor_id": "0", "x": 0.0, "y": 0.5, "w": 0.2, "h": 0.1}


def make_window(config: Config, calls: list | None = None) -> ControlWindow:
    calls = calls if calls is not None else []
    return ControlWindow(
        config,
        lambda: calls.append("start"),
        lambda: calls.append("stop"),
        lambda: calls.append("add"),
        lambda index: calls.append(("adjust", index)),
        lambda index: calls.append(("delete", index)),
        lambda cfg: calls.append("apply"),
        dict,
        Rect(0, 0, 1000, 1000),
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
    window = make_window(Config(regions=[ONE]), calls)
    window.start_button.click()
    assert calls == ["start"]
    window.close()


def test_start_is_blocked_without_regions(qapp):
    calls: list = []
    window = make_window(Config(regions=[]), calls)
    assert window.start_button.isEnabled() is False
    window.start_button.click()
    assert calls == []
    window.close()


def test_stop_button_calls_stop_when_running(qapp):
    calls: list = []
    window = make_window(Config(regions=[ONE]), calls)
    window.start_button.click()
    window.start_button.click()
    assert calls == ["start", "stop"]
    window.close()


def test_rows_follow_the_saved_regions(qapp):
    cfg = Config(region_mode="multi", regions=[ONE, TWO])
    window = make_window(cfg)
    assert window.region_list.count() == 2
    assert window.region_list.item(0).text() == "1  ·  500 × 100"
    window.close()


def test_selecting_a_row_enables_adjust_and_delete(qapp):
    window = make_window(Config(region_mode="multi", regions=[ONE, TWO]))
    window.region_list.setCurrentRow(1)
    assert window.selected_region() == 1
    assert window.adjust_button.isEnabled() and window.delete_button.isEnabled()
    window.close()


def test_add_calls_the_add_callback(qapp):
    calls: list = []
    window = make_window(Config(regions=[ONE]), calls)
    window.add_button.click()
    assert calls == ["add"]
    window.close()


def test_adjust_passes_the_selected_index(qapp):
    calls: list = []
    window = make_window(Config(region_mode="multi", regions=[ONE, TWO]), calls)
    window.region_list.setCurrentRow(1)
    window.adjust_button.click()
    assert ("adjust", 1) in calls
    window.close()


def test_delete_passes_the_selected_index(qapp):
    calls: list = []
    window = make_window(Config(region_mode="multi", regions=[ONE, TWO]), calls)
    window.region_list.setCurrentRow(1)
    window.delete_button.click()
    assert ("delete", 1) in calls
    window.close()


def test_delete_is_disabled_with_one_region(qapp):
    window = make_window(Config(regions=[ONE]))
    window.region_list.setCurrentRow(0)
    assert window.delete_button.isEnabled() is False
    window.close()


def test_single_mode_marks_extra_rows_inactive_without_deleting_them(qapp):
    window = make_window(Config(region_mode="single", regions=[ONE, TWO]))
    assert window.region_list.count() == 2
    assert "inactive" in window.region_list.item(1).text()
    window.close()


def test_read_config_persists_the_appearance(qapp):
    window = make_window(Config())
    window.set_appearance(
        text_color="#FFE08A", bg_color="#000000", font_family="Consolas", min_font_px=14
    )
    cfg = window.read_config()
    assert (cfg.text_color, cfg.bg_color, cfg.font_family, cfg.min_font_px) == (
        "#FFE08A",
        "#000000",
        "Consolas",
        14,
    )
    window.close()


def test_theme_is_applied_once(qapp):
    from ocrtrans.ui import theme

    theme.apply(qapp)
    assert theme.ACCENT in qapp.styleSheet()


def test_appearance_change_is_applied_immediately(qapp):
    calls: list = []
    window = make_window(Config(regions=[ONE]), calls)
    window.min_font.setValue(20)
    assert "apply" in calls
    window.close()


def test_the_debug_checkbox_is_off_by_default(qapp, monkeypatch):
    monkeypatch.delenv("OCRTRANS_DEBUG", raising=False)
    window = make_window(Config(regions=[ONE]))
    assert window.debug.isChecked() is False
    window.close()


def test_the_debug_checkbox_sets_the_log_level(qapp):
    import logging

    root = logging.getLogger()
    before = root.level
    try:
        window = make_window(Config(regions=[ONE]))
        window.debug.setChecked(True)
        assert root.level == logging.DEBUG
        window.debug.setChecked(False)
        assert root.level == logging.INFO
        window.close()
    finally:
        root.setLevel(before)


def test_the_window_hints_when_no_region_exists(qapp):
    window = make_window(Config(regions=[]))
    assert not window.region_hint.isHidden()
    window.close()
