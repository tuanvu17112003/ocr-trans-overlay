"""Configuration load/save with validation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENGINES = frozenset({"nllb", "envit5", "argos", "google", "gemini", "deepl", "mymemory"})
STABILITY_PROFILES = frozenset({"safe", "fast"})
REGION_MODES = frozenset({"single", "multi"})


@dataclass
class Config:
    engine: str = "nllb"
    stability_profile: str = "safe"
    capture_fps: int = 15
    region_mode: str = "single"
    regions: list[dict] = field(default_factory=list)
    min_font_px: int = 12
    font_family: str = "Segoe UI"
    text_color: str = "#FFFFFF"
    bg_color: str = "#101014"
    min_confidence: float = 0.6
    ocr_downscale: int = 1280
    keys: dict[str, str] = field(default_factory=dict)
    fallback_to_local: bool = True
    online_consent: bool = False


def default_config_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "OCRTrans" / "config.json"


def _validate(cfg: Config) -> None:
    if cfg.engine not in ENGINES:
        raise ValueError(f"engine: {cfg.engine!r} is not one of {sorted(ENGINES)}")
    if cfg.stability_profile not in STABILITY_PROFILES:
        raise ValueError(f"stability_profile: {cfg.stability_profile!r} must be safe or fast")
    if not 5 <= cfg.capture_fps <= 30:
        raise ValueError(f"capture_fps: {cfg.capture_fps} must be between 5 and 30")
    if cfg.region_mode not in REGION_MODES:
        raise ValueError(f"region_mode: {cfg.region_mode!r} must be one of {sorted(REGION_MODES)}")
    if cfg.min_font_px < 8:
        raise ValueError(f"min_font_px: {cfg.min_font_px} must be at least 8")
    if not 0.0 < cfg.min_confidence <= 1.0:
        raise ValueError(f"min_confidence: {cfg.min_confidence} must be in (0, 1]")


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    if not path.exists():
        return Config()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("region_mode") == "monitor":
        raw["region_mode"] = "single"  # the mode is gone; keep the saved regions
    known = {f for f in Config.__dataclass_fields__}
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    _validate(cfg)
    return cfg


def save_config(cfg: Config, path: Path | None = None) -> None:
    _validate(cfg)
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
