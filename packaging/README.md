# Packaging

Builds the portable Windows executable that is attached to a GitHub Release.

## What the release contains

`OCRTrans-<version>-win64-cpu.zip` — unpack anywhere and run `OCRTrans.exe`.
No Python, no installer, no NVIDIA driver required.

| Requirement | Notes |
|---|---|
| Windows 10/11 x64 | Same machine class as the source build |
| No Python | Everything is bundled, including the Qt runtime |
| No GPU | OCR runs on the onnxruntime CPU provider and translation uses Argos |

The first Start still downloads the Argos en->vi language pack (~100 MB) into
`%USERPROFILE%\.local\share\argos-translate`. Everything else (RapidOCR's
ONNX models, the Qt runtime) is inside the zip.

**Not included on purpose:** the CUDA build. `torch==2.14.1+cu130` puts ~2.9 GB
of CUDA kernels in `torch/lib`, well past the 2 GB per-file limit a GitHub
Release accepts, and a machine with an NVIDIA GPU installs faster with
`install.ps1` than by downloading a bundle it cannot share. The frozen build is
the CPU-only path; GPU users keep the source install.

## Build it yourself

```powershell
py -3.13 -m venv .build-venv
.\.build-venv\Scripts\python.exe -m pip install -r requirements-cpu.txt "pyinstaller==6.16.0"
.\scripts\build_exe.ps1 -Clean -Zip
```

Output:

| Path | Size | What |
|---|---|---|
| `dist\OCRTrans\` | ~930 MB | Onedir build; this is what the zip contains |
| `dist\OCRTrans-<ver>-win64-cpu.zip` | ~350 MB | The release artifact |
| `dist\OCRTrans-<ver>-win64-cpu.zip.sha256` | 64 B | Checksum for the release notes |

## Why onedir and not onefile

Onefile re-extracts ~930 MB into `%TEMP%\_MEI*` on every launch: the first
window takes tens of seconds and any leftover folder breaks the next start. Onedir
launches immediately and keeps the deps inspectable when a DLL is missing.

## Notes for whoever rebuilds

- `packaging/ocrtrans.spec` cannot exclude `torch._inductor`, `torch._dynamo` or
  `torch.distributed`: `torch/__init__.py` imports them at module level and a
  failure there sets a process-wide flag (`sys._torch_import_started`) that
  makes every later `import torch` raise. Removing ~300 MB of CUDA kernels from
  the bundle is done by building in a CPU-only venv, not by excluding modules.
- The RapidOCR models are data files inside the wheel; `collect_data_files`
  picks them up. If a future RapidOCR moves them out of the package, the
  build will fail at the first OCR pass with "model not found" rather than at
  build time — check `dist\OCRTrans\_internal\rapidocr\models`.
- Windowed builds have `sys.stdout`/`sys.stderr` set to `None`;
  `packaging/ocrtrans_exe.py` replaces them with a devnull sink before the app
  starts, otherwise the logging handler raises.
