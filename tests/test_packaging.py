import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _lines(name):
    raw = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in raw if line.strip() and not line.startswith("#")]


def test_gpu_requirements_keep_the_cuda_index():
    lines = _lines("requirements.txt")
    assert lines[0].startswith("--extra-index-url https://download.pytorch.org/whl/cu")
    assert any(line.startswith("torch==") and "+cu" in line for line in lines)
    assert any(line.startswith("onnxruntime-gpu==") for line in lines)


def test_cpu_requirements_have_no_cuda_and_no_transformers():
    lines = _lines("requirements-cpu.txt")
    assert not any(line.startswith("--extra-index-url") for line in lines)
    assert any(line.startswith("onnxruntime==") for line in lines)
    assert not any("onnxruntime-gpu" in line for line in lines)
    assert not any(line.startswith("transformers") for line in lines)
    assert all("+cu" not in line for line in lines)


def test_every_requirement_is_pinned():
    for name in ("requirements.txt", "requirements-cpu.txt"):
        for line in _lines(name):
            if line.startswith("--"):
                continue
            assert "==" in line, f"{name}: {line} is not pinned"


def test_installer_parses():
    script = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert "-Cpu" in script and "-DryRun" in script
    assert '"-m", "pip", "install", "-e", ".", "--no-deps"' in script


def test_installer_is_valid_powershell():
    if shutil.which("powershell") is None:
        pytest.skip("no PowerShell on this machine")
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$null = [scriptblock]::Create((Get-Content -Raw -LiteralPath 'install.ps1'))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
