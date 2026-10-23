"""Free online translation providers and the degraded-mode router."""

from __future__ import annotations

import json
import time
from typing import Literal

import requests

from .translate import Engine

GOOGLE_URL = "https://translate.googleapis.com/translate_a/single"
MYMEMORY_URL = "https://api.mymemory.translated.net/get"
DEEPL_URL = "https://api-free.deepl.com/v2/translate"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class ProviderError(Exception):
    pass


class QuotaError(ProviderError):
    pass


def _check(resp) -> None:
    if resp.status_code in (401, 429):
        raise QuotaError(f"provider quota/auth error: {resp.status_code}")
    if resp.status_code >= 400:
        raise ProviderError(f"provider error: {resp.status_code}")


class _OnlineBase:
    kind: Literal["online"] = "online"

    def __init__(self, session: requests.Session, timeout: float = 3.0) -> None:
        self.session = session
        self.timeout = timeout

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass


class GoogleEngine(_OnlineBase):
    name = "google"

    def translate(self, texts: list[str]) -> list[str]:
        out = []
        for text in texts:
            try:
                resp = self.session.get(
                    GOOGLE_URL,
                    params={"client": "gtx", "sl": "en", "tl": "vi", "dt": "t", "q": text},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise ProviderError(str(exc)) from exc
            _check(resp)
            payload = resp.json()
            segments = payload[0] or []
            out.append("".join(seg[0] for seg in segments if seg and seg[0]))
        return out


class MyMemoryEngine(_OnlineBase):
    name = "mymemory"

    def translate(self, texts: list[str]) -> list[str]:
        out = []
        for text in texts:
            try:
                resp = self.session.get(
                    MYMEMORY_URL,
                    params={"q": text, "langpair": "en|vi"},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise ProviderError(str(exc)) from exc
            _check(resp)
            out.append(resp.json()["responseData"]["translatedText"])
        return out


class DeepLEngine(_OnlineBase):
    name = "deepl"

    def __init__(self, api_key: str, session: requests.Session, timeout: float = 3.0) -> None:
        super().__init__(session, timeout)
        self.api_key = api_key

    def translate(self, texts: list[str]) -> list[str]:
        try:
            resp = self.session.post(
                DEEPL_URL,
                headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                json={"text": texts, "source_lang": "EN", "target_lang": "VI"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError(str(exc)) from exc
        _check(resp)
        return [item["text"] for item in resp.json()["translations"]]


class GeminiEngine(_OnlineBase):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        session: requests.Session,
        model: str = "gemini-2.5-flash",
        timeout: float = 6.0,
    ) -> None:
        super().__init__(session, timeout)
        self.api_key = api_key
        self.model = model

    def translate(self, texts: list[str]) -> list[str]:
        prompt = (
            "Translate every string in this JSON array to Vietnamese. "
            "Reply with only a JSON array of the same length, no commentary: "
            + json.dumps(texts, ensure_ascii=False)
        )
        try:
            resp = self.session.post(
                GEMINI_URL.format(model=self.model),
                params={"key": self.api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError(str(exc)) from exc
        _check(resp)
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        result = json.loads(text)
        if not isinstance(result, list) or len(result) != len(texts):
            raise ProviderError("gemini returned a malformed array")
        return [str(item) for item in result]


class OnlineRouter:
    """Primary online engine, cooldown on quota errors, local fallback."""

    kind: Literal["online"] = "online"

    def __init__(
        self,
        primary: Engine,
        local_fallback: Engine | None,
        cooldown_s: float = 60.0,
    ) -> None:
        self.primary = primary
        self.local_fallback = local_fallback
        self.cooldown_s = cooldown_s
        self._cooldown_until = 0.0

    @property
    def name(self) -> str:
        return self.primary.name

    def load(self) -> None:
        pass

    def unload(self) -> None:
        for engine in (self.primary, self.local_fallback):
            if engine is not None:
                engine.unload()

    @property
    def healthy(self) -> bool:
        return time.monotonic() >= self._cooldown_until

    def _fallback(self, text: str) -> tuple[str, bool]:
        if self.local_fallback is None:
            raise ProviderError("online provider unavailable and no local fallback")
        return self.local_fallback.translate([text])[0], True

    def translate(self, text: str) -> tuple[str, bool]:
        if not self.healthy:
            return self._fallback(text)
        try:
            return self.primary.translate([text])[0], False
        except QuotaError:
            self._cooldown_until = time.monotonic() + self.cooldown_s
            return self._fallback(text)
        except ProviderError:
            time.sleep(0.2)
            try:
                return self.primary.translate([text])[0], False
            except ProviderError:
                return self._fallback(text)
