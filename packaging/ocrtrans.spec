# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the portable Windows build.

Only the CPU flavour is produced here, on purpose:

  * the CUDA torch wheel is ~2.9 GB on disk (torch/lib alone 456 MB of cuBLAS
    plus 403 MB torch_cuda), so the resulting folder is well past the 2 GB
    per-file limit a GitHub Release accepts, and
  * machines that own an NVIDIA GPU already install the CUDA build with
    install.ps1, which is smaller for them than a bundle they cannot share.

The CPU bundle still ships the Argos engine (torch CPU comes in through
stanza) so a fresh machine translates offline with no NVIDIA driver.

Build it with scripts/build_exe.ps1; that script owns the build venv.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
ICON = ROOT / "packaging" / "ocrtrans.ico"

# --- data files -----------------------------------------------------------
datas = []
# RapidOCR resolves its ONNX models and yaml configs relative to the package.
datas += collect_data_files("rapidocr", include_py_files=False)
# Argos needs its settings/module metadata; language packs are downloaded into
# %USERPROFILE% on first use and are deliberately not bundled.
datas += collect_data_files("argostranslate", include_py_files=False)

# --- hidden imports -------------------------------------------------------
# The engines are imported inside functions; the onnxruntime providers and the
# native capture module are loaded by name at runtime.
hiddenimports = [
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._pybind_state",
    "windows_capture",
    "rapidocr",
    "argostranslate",
    "argostranslate.package",
    "argostranslate.translate",
    "ctranslate2",
    "stanza",
    "sentencepiece",
    "PyQt6.sip",
]
hiddenimports += collect_submodules("rapidocr.inference_engine")

# --- exclusions -----------------------------------------------------------
# Torch is reached through argostranslate -> stanza. Its submodules cannot be
# excluded: torch/__init__.py imports torch._inductor, torch._dynamo and
# torch.distributed at module level, and a failure there sets a process-wide
# flag that makes every later `import torch` raise. Only genuinely unused
# third-party packages are excluded here.
excludes = [
    "tkinter",
    "matplotlib",
    "pytest",
    "pandas",
    "scipy",
    "IPython",
    "notebook",
    "PyQt6.QtWebEngineCore",
    "PyQt6.QtWebEngineWidgets",
    "PyQt6.Qt3DCore",
    "PyQt6.QtMultimedia",
    "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtBluetooth",
    "PyQt6.QtNfc",
    "PyQt6.QtPositioning",
    "PyQt6.QtSerialPort",
    "PyQt6.QtTest",
    "PyQt6.QtDesigner",
    "PyQt6.QtHelp",
    "PyQt6.QtPdf",
    "PyQt6.QtPdfWidgets",
    "PyQt6.QtQuick",
    "PyQt6.QtQml",
    "PyQt6.QtOpenGL",
    "PyQt6.QtDBus",
    "PyQt6.QtSql",
]

a = Analysis(
    [str(ROOT / "packaging" / "ocrtrans_exe.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OCRTrans",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OCRTrans",
)