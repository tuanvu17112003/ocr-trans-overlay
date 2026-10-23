"""Translation engines (local NLLB / EnViT5), cache and service."""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Protocol

MAX_NEW_TOKENS = 96

log = logging.getLogger(__name__)


class Engine(Protocol):
    name: str
    kind: Literal["local", "online"]

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def translate(self, texts: list[str]) -> list[str]: ...


def normalize(text: str) -> str:
    """Trim, drop control characters, collapse whitespace."""
    cleaned = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return " ".join(cleaned.split())


class TranslationCache:
    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        self._data: OrderedDict[str, str] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> str | None:
        if key not in self._data:
            self._misses += 1
            return None
        self._hits += 1
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {"size": len(self._data), "hits": self._hits, "misses": self._misses}


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch is optional for online-only use
        pass


class _LocalEngine:
    name = "local"
    kind: Literal["local"] = "local"
    _prefix = ""
    _strip_prefix = ""
    _src_lang: str | None = None
    _tgt_lang: str | None = None

    def __init__(self, model_dir: Path, device: str = "cuda") -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model directory not found: {self.model_dir}")
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        kwargs = {"src_lang": self._src_lang} if self._src_lang else {}
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), **kwargs)
        dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
        if dtype is torch.float16:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_dir), torch_dtype=dtype
            ).to(self.device)
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir)).to(self.device)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        _empty_cuda_cache()

    def translate(self, texts: list[str]) -> list[str]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"{self.name} engine is not loaded")
        import torch

        inputs = self._tokenizer(
            [self._prefix + t for t in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        extra = {}
        if self._tgt_lang:
            extra["forced_bos_token_id"] = self._tokenizer.convert_tokens_to_ids(self._tgt_lang)
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **extra)
        out = self._tokenizer.batch_decode(ids, skip_special_tokens=True)
        if self._strip_prefix:
            out = [t.removeprefix(self._strip_prefix).strip() for t in out]
        return out


class NllbEngine(_LocalEngine):
    name = "nllb"
    _src_lang = "eng_Latn"
    _tgt_lang = "vie_Latn"


class Envit5Engine(_LocalEngine):
    name = "envit5"
    _prefix = "en: "
    _strip_prefix = "vi: "


class TranslationService:
    def __init__(
        self,
        engine: Engine,
        cache: TranslationCache,
        fallback: Engine | None = None,
    ) -> None:
        self.engine = engine
        self.cache = cache
        self.fallback = fallback
        self._lock = threading.Lock()
        self._in_flight = 0
        self.degraded = False
        self.errors = 0

    def translate(self, text: str) -> tuple[str, bool]:
        key = normalize(text)
        if not key:
            return "", False
        cached = self.cache.get(key)
        if cached is not None:
            self.degraded = False
            return cached, True

        with self._lock:
            self._in_flight += 1
        try:
            try:
                result = self._call(self.engine, key)
            except Exception:  # noqa: BLE001 - online failure falls back to local
                self._count_error()
                if self.fallback is None:
                    raise
                result = self._call(self.fallback, key)
        finally:
            with self._lock:
                self._in_flight -= 1

        result = result.strip()
        self.cache.put(key, result)
        return result, False

    def _call(self, engine: Engine, text: str) -> str:
        out = engine.translate([text])
        if isinstance(out, tuple):  # OnlineRouter returns (text, degraded)
            self.degraded = bool(out[1])
            return out[0]
        self.degraded = False
        return out[0]

    def translate_many(self, texts: list[str]) -> list[tuple[str, bool]]:
        """Translate a batch; one engine call for the cache misses."""
        keys = [normalize(text) for text in texts]
        results: list[tuple[str, bool] | None] = []
        missing: list[str] = []
        for key in keys:
            if not key:
                results.append(("", False))
                continue
            cached = self.cache.get(key)
            if cached is not None:
                results.append((cached, True))
                continue
            results.append(None)
            if key not in missing:
                missing.append(key)

        mapping: dict[str, str] = {}
        if missing:
            translated = self._translate_batch(missing)
            for key, value in zip(missing, translated):
                value = value.strip()
                mapping[key] = value
                if value:
                    self.cache.put(key, value)

        return [
            result if result is not None else (mapping.get(key, ""), False)
            for key, result in zip(keys, results)
        ]

    def _translate_batch(self, texts: list[str]) -> list[str]:
        with self._lock:
            self._in_flight += 1
        try:
            try:
                return self._call_batch(self.engine, texts)
            except Exception:  # noqa: BLE001 - online failure falls back to local
                self._count_error()
                if self.fallback is None:
                    return [""] * len(texts)
                return self._call_batch(self.fallback, texts)
        finally:
            with self._lock:
                self._in_flight -= 1

    def _count_error(self) -> None:
        with self._lock:
            self.errors += 1
        if self.errors == 1 or self.errors % 100 == 0:
            log.exception("translation failed (%d so far)", self.errors)

    def release(self, timeout: float = 5.0) -> bool:
        """Unload the current engine once no translation is in flight."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if not self._in_flight:
                    self.engine.unload()
                    _empty_cuda_cache()
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def adopt(self, engine: Engine) -> None:
        """Install an already-loaded engine and drop the old cache."""
        with self._lock:
            self.engine = engine
            self.cache = TranslationCache()

    def _call_batch(self, engine: Engine, texts: list[str]) -> list[str]:
        if hasattr(engine, "primary"):  # OnlineRouter only accepts a single text
            out: list[str] = []
            for text in texts:
                result = engine.translate(text)
                if isinstance(result, tuple):
                    self.degraded = bool(result[1])
                    out.append(result[0])
                else:
                    out.append(result[0])
            return out
        translated = engine.translate(texts)
        self.degraded = False
        return [text.strip() for text in translated]

    def switch(self, engine: Engine) -> None:
        """Swap engines; the caller passes a loaded engine."""
        with self._lock:
            if self._in_flight:
                raise RuntimeError("engine busy")
            old = self.engine
            if old is not engine:
                old.unload()
                _empty_cuda_cache()
            self.engine = engine

