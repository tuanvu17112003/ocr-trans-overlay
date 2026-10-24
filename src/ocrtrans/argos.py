"""Optional offline Argos English->Vietnamese engine (installs its model once)."""

from __future__ import annotations

import threading


class ArgosEngine:
    name = "argos"
    kind = "local"

    def __init__(self, from_code: str = "en", to_code: str = "vi") -> None:
        self.from_code = from_code
        self.to_code = to_code
        self.installed_translation = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self.installed_translation is not None:
                return
            try:
                import argostranslate.package as package
                import argostranslate.translate as translate
            except ImportError as exc:
                raise RuntimeError(
                    "argostranslate is not usable "
                    f"({exc}). Install it with: pip install --no-deps "
                    "argostranslate ctranslate2 stanza==1.10.1 minisbd emoji"
                ) from exc

            translation = self._resolve(translate)
            if translation is None:
                try:
                    package.update_package_index()
                    available = package.get_available_packages()
                    wanted = next(
                        (
                            item
                            for item in available
                            if item.from_code == self.from_code
                            and item.to_code == self.to_code
                        ),
                        None,
                    )
                    if wanted is None:
                        raise RuntimeError(
                            f"Argos has no {self.from_code}->{self.to_code} package"
                        )
                    wanted.install()
                except Exception as exc:  # noqa: BLE001 - index/install failures
                    raise RuntimeError(
                        f"could not install the Argos {self.from_code}->{self.to_code} "
                        f"model: {exc}"
                    ) from exc
                translation = self._resolve(translate)
            if translation is None:
                raise RuntimeError(
                    f"Argos {self.from_code}->{self.to_code} model is not available"
                )
            self.installed_translation = translation

    def _resolve(self, translate):
        try:
            return translate.get_translation_from_codes(self.from_code, self.to_code)
        except (AttributeError, StopIteration):
            # the real API raises when no language pack is installed yet
            return None

    def unload(self) -> None:
        with self._lock:
            self.installed_translation = None

    def translate(self, texts: list[str]) -> list[str]:
        if self.installed_translation is None:
            raise RuntimeError("argos engine is not loaded")
        return [self.installed_translation.translate(text) for text in texts]
