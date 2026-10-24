"""Feasibility gate for the realtime OCR translation overlay.

Standalone: does not import ocrtrans. Prints one row per check and exits
non-zero unless the overall status is PASS or PASS_WITH_FALLBACK.

Statuses: PASS, PASS_WITH_FALLBACK, BLOCKED, FAIL.
"""

from __future__ import annotations

import ctypes
import statistics
import sys
import time
import traceback

ROWS: list[tuple[str, str, float | None, str]] = []

WDA_EXCLUDEFROMCAPTURE = 0x00000011


def _prepare_dlls() -> None:
    """onnxruntime-gpu needs the CUDA 13 DLLs that ship inside torch/lib."""
    import os
    import pathlib

    try:
        import torch

        os.add_dll_directory(str(pathlib.Path(torch.__file__).parent / "lib"))
    except Exception:  # noqa: BLE001 - absence is reported by the checks
        pass


_prepare_dlls()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def record(name: str, status: str, ms: float | None = None, detail: str = "") -> None:
    ROWS.append((name, status, ms, detail))
    ms_s = "-" if ms is None else f"{ms:.1f}"
    print(f"{name:<24} | {ms_s:>8} | {detail:<48} | {status}")


def check(name: str, fn):
    started = time.perf_counter()
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a gate reports, it does not crash
        record(name, "FAIL", (time.perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)
        return None
    record(name, status, (time.perf_counter() - started) * 1000, detail)
    return status


def check_imports():
    import cv2  # noqa: F401
    import onnxruntime  # noqa: F401
    import PyQt6.QtWidgets  # noqa: F401
    import rapidocr  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import windows_capture  # noqa: F401

    return "PASS", "all packages import"


def check_ort_cuda():
    import pathlib

    import onnxruntime as ort
    import rapidocr

    model = pathlib.Path(rapidocr.__file__).parent / "models" / "PP-OCRv6_det_small.onnx"
    session = ort.InferenceSession(str(model), providers=["CUDAExecutionProvider"])
    providers = session.get_providers()
    if "CUDAExecutionProvider" in providers:
        return "PASS", f"session providers: {providers}"
    return "BLOCKED", f"session fell back to {providers}"


def check_dxgi():
    import numpy as np
    from windows_capture import DxgiDuplicationSession

    session = DxgiDuplicationSession()
    deadline = time.perf_counter() + 3.0
    best = None
    while time.perf_counter() < deadline:
        frame = session.acquire_frame(timeout_ms=500)
        if frame is None:
            continue
        arr = frame.to_bgr()          # HDR screens return rgba16f; to_bgr tonemaps
        variance = float(np.var(arr))
        best = (arr.shape[1], arr.shape[0], variance)
        if variance >= 1.0:
            return "PASS", f"{best[0]}x{best[1]} var={variance:.0f}"
        time.sleep(0.1)
    if best is None:
        return "BLOCKED", "no DXGI frame within 3s"
    return "BLOCKED", f"frames stayed uniform ({best[0]}x{best[1]} var={best[2]:.1f})"


def check_wgc():
    from windows_capture import WindowsCapture

    got: list[float] = []
    capture = WindowsCapture(monitor_index=1, cursor_capture=False, draw_border=False)

    @capture.event
    def on_frame_arrived(frame, control):
        got.append(time.perf_counter())
        control.stop()

    @capture.event
    def on_closed():
        pass

    started = time.perf_counter()
    capture.start()
    elapsed = time.perf_counter() - started
    if not got:
        return "BLOCKED", "no WGC frame"
    if elapsed > 2.0:
        return "PASS_WITH_FALLBACK", f"WGC slow: {elapsed * 1000:.0f}ms"
    return "PASS", f"first frame {elapsed * 1000:.0f}ms"


def make_text_image():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (800, 200), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 44)
    draw.text((30, 70), "Open the gate, traveler.", font=font, fill=(240, 240, 240))
    return np.array(img)[:, :, ::-1].copy()


def check_ocr(use_cuda: bool):
    from rapidocr import RapidOCR

    params = {"EngineConfig.onnxruntime.use_cuda": use_cuda}
    engine = RapidOCR(params=params)
    image = make_text_image()
    engine(image, use_det=True, use_cls=False, use_rec=True)  # warmup
    times, texts = [], []
    for _ in range(3):
        started = time.perf_counter()
        out = engine(image, use_det=True, use_cls=False, use_rec=True)
        times.append((time.perf_counter() - started) * 1000)
        texts = list(out.txts) if out.txts else []
    got = " ".join(texts)
    import difflib

    ratio = difflib.SequenceMatcher(None, got.lower(), "open the gate, traveler.").ratio()
    if ratio < 0.8:
        return "FAIL", f"OCR mismatch: {got!r}"
    return "PASS", f"{statistics.median(times):.0f}ms, text={got!r}"


def check_mt(model: str, prefix: str, src_lang: str | None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = (
        AutoTokenizer.from_pretrained(model, src_lang=src_lang)
        if src_lang
        else AutoTokenizer.from_pretrained(model)
    )
    net = AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype=torch.float16).to("cuda")
    text = prefix + "Open the gate, traveler."
    inputs = tokenizer([text], return_tensors="pt", padding=True).to("cuda")
    extra = {}
    if src_lang:
        extra["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids("vie_Latn")
    times = []
    out_text = ""
    for _ in range(3):
        started = time.perf_counter()
        with torch.no_grad():
            ids = net.generate(**inputs, max_new_tokens=96, **extra)
        times.append((time.perf_counter() - started) * 1000)
        out_text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    peak = torch.cuda.max_memory_allocated() / 1e9
    del net
    torch.cuda.empty_cache()
    status = "PASS" if peak <= 2.0 and out_text.strip() else "BLOCKED"
    return status, f"{statistics.median(times):.0f}ms peak={peak:.2f}GB out={out_text[:40]!r}"


def check_self_capture():
    import numpy as np
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QWidget
    from windows_capture import DxgiDuplicationSession

    app = QApplication.instance() or QApplication(sys.argv)
    win = QWidget()
    win.setStyleSheet("background: #FF00FF;")
    win.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
    win.setGeometry(120, 120, 320, 200)
    win.show()
    win.raise_()
    for _ in range(10):
        app.processEvents()
        time.sleep(0.05)

    hwnd = int(win.winId())
    session = DxgiDuplicationSession()

    def grab_count_magenta(seconds: float = 1.2, mode: str = "max") -> float:
        best = -1.0
        last = -1.0
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            frame = session.acquire_frame(timeout_ms=300)
            if frame is None:
                continue
            img = frame.to_bgr()
            magenta = (img[:, :, 0] > 200) & (img[:, :, 2] > 200) & (img[:, :, 1] < 60)
            last = float(magenta.mean())
            best = max(best, last)
            time.sleep(0.1)
        return last if mode == "last" else best

    before = grab_count_magenta()
    ok = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    for _ in range(10):
        app.processEvents()
        time.sleep(0.1)
    after = grab_count_magenta(mode="last")
    win.close()
    app.processEvents()

    if before < 0 or after < 0:
        return "BLOCKED", "could not grab a frame"
    if before < 0.002:
        return "BLOCKED", f"control capture failed (magenta {before:.2%})"
    if not ok:
        return "BLOCKED", "SetWindowDisplayAffinity returned 0"
    if after < 0.0002:
        return "PASS", f"magenta before {before:.2%} after {after:.3%}"
    return "BLOCKED", f"overlay still captured (magenta {after:.2%})"


def main() -> int:
    print(f"{'check':<24} | {'ms':>8} | {'detail':<48} | status")
    check("imports", check_imports)
    check("ort-cuda", check_ort_cuda)
    check("dxgi", check_dxgi)
    check("wgc", check_wgc)

    ocr_gpu = check("ocr-gpu", lambda: check_ocr(True))
    if ocr_gpu != "PASS":
        check("ocr-cpu", lambda: check_ocr(False))

    check("mt-nllb", lambda: check_mt("facebook/nllb-200-distilled-600M", "", "eng_Latn"))
    check("mt-envit5", lambda: check_mt("VietAI/envit5-translation", "en: ", None))
    check("self-capture", check_self_capture)

    worst = "PASS"
    order = ["PASS", "PASS_WITH_FALLBACK", "FAIL", "BLOCKED"]
    for _, status, _, _ in ROWS:
        if order.index(status) > order.index(worst):
            worst = status
    print(f"\nOverall: {worst}")
    return 0 if worst in ("PASS", "PASS_WITH_FALLBACK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
