# Feasibility Gate — 2026-10-23

Overall: **PASS**

Machine: RTX 3080 10 GB, Windows 11 25H2 (26200), Python 3.13.12, display 2560x1440 HDR.
Command: `.\.venv\Scripts\python scripts\feasibility.py` (exit 0).

| check | ms | detail | status |
|-------|----|--------|--------|
| imports | 1209 | all packages import | PASS |
| ort-cuda | 198 | session providers: CUDAExecutionProvider, CPUExecutionProvider | PASS |
| dxgi | 253 | 2560x1440 var=1267 | PASS |
| wgc | 228 | first frame 228 ms | PASS |
| ocr-gpu | 1565 | 118 ms per 800x200 image (median of 3) | PASS |
| mt-nllb | 7003 | 202 ms, peak 1.25 GB, output `Mở cổng đi, người du lịch.` | PASS |
| mt-envit5 | 4373 | 295 ms, peak 0.64 GB, output `vi: Mở cổng đi, khách du lịch.` | PASS |
| self-capture | 3842 | magenta before 1.74%, after 0.000% | PASS |

Repeated run (warmer caches): OCR 182 ms, NLLB 216 ms, EnViT5 267 ms, self-capture unchanged.

## Findings that changed the plan

1. **PyPI `torch` on Windows is CPU-only.** Pinned `torch==2.14.1+cu130` from the
   PyTorch CUDA index. `torch.cuda.is_available()` is True, CUDA 13.0.
2. **transformers 5.20 cannot load EnViT5.** Its `tokenizer.json` fails with
   `'dict' object is not an instance of 'Sequence'` and the model loses weight
   tying (empty output). Pinned `transformers<5` (4.57.6); both engines work.
3. **onnxruntime-gpu needs CUDA 13 DLLs that ship in `torch/lib`.** Without
   `os.add_dll_directory(<torch>/lib)` the CUDA provider silently falls back to
   CPU. The check now opens a real CUDA session and asserts the provider.
4. **The display is HDR.** DXGI returns `rgba16f`; `to_numpy()` gives zeros/NaN.
   `to_bgr()` tonemaps correctly and is the only capture path.
5. **EnViT5 output carries a `vi: ` prefix** that must be stripped.
6. **Measured translation latency is above the original budget:** NLLB ~210 ms
   (budget 150), EnViT5 ~270-330 ms (budget 250), OCR 118-182 ms (budget 60).
   Budgets will be re-ruled after the OCR downscale/tuning task, or relaxed with
   the measurement recorded here.
7. WGC first frame takes ~230 ms because the session starts on `start()`; this is
   startup cost, not per-frame latency.
8. VRAM is fine: 1.25 GB (NLLB) and 0.64 GB (EnViT5), under the 2 GB ceiling.

## Fallbacks exercised

None needed. Affinity exclusion works (1.74% -> 0.000%), so the hide-gate is not
the primary path; it stays implemented and tested as a fallback.
