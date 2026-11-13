# Build the portable Windows EXE for OCRTrans.
#
#   .\scripts\build_exe.ps1              # CPU build (default), -> dist\OCRTrans
#   .\scripts\build_exe.ps1 -Clean       # wipe build/ and dist/ first
#   .\scripts\build_exe.ps1 -Zip         # also produce dist\OCRTrans-<ver>-win64.zip
#
# The build venv (.build-venv) is CPU-only and separate from .venv: mixing the
# CUDA torch wheel (2.9 GB) into the bundle would blow past the GitHub Release
# 2 GB per-file limit, and the portable build is meant for machines without an
# NVIDIA driver anyway. GPU users keep install.ps1.
param(
    [switch]$Clean,
    [switch]$Zip,
    [string]$Name = "OCRTrans"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venv = Join-Path $root ".build-venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "ERROR: build venv not found. Run:" -ForegroundColor Red
    Write-Host "  py -3.13 -m venv .build-venv"
    Write-Host "  .\.build-venv\Scripts\python.exe -m pip install -r requirements-cpu.txt pyinstaller==6.16.0"
    exit 1
}

if ($Clean) {
    foreach ($dir in @("build", "dist")) {
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    }
}

$version = & $py -c "import sys; sys.path.insert(0,'src'); import ocrtrans; print(ocrtrans.__version__)"
$version = $version.Trim()
Write-Host ">> building OCRTrans $version ($Name)" -ForegroundColor Cyan

& $py -m PyInstaller --noconfirm --clean --distpath "dist" --workpath "build" "packaging\ocrtrans.spec"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyInstaller failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

$out = Join-Path $root "dist\$Name"
if (-not (Test-Path (Join-Path $out "OCRTrans.exe"))) {
    Write-Host "ERROR: dist\$Name\OCRTrans.exe missing after the build" -ForegroundColor Red
    exit 1
}

$sizeMb = [math]::Round(((Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB, 0)
Write-Host ">> built: $out ($sizeMb MB)" -ForegroundColor Green

if ($Zip) {
    $stage = Join-Path $root "build\release\$Name"
    if (Test-Path (Split-Path $stage -Parent)) { Remove-Item -Recurse -Force (Split-Path $stage -Parent) }
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    Copy-Item -Path (Join-Path $out "*") -Destination $stage -Recurse -Force
    # User-facing notes live outside the executable so they survive a rebuild.
    $notes = Join-Path $root "packaging\RELEASE_README.txt"
    if (Test-Path $notes) {
        (Get-Content $notes -Raw).Replace('{VERSION}', $version) |
            Set-Content (Join-Path $stage "README.txt") -Encoding UTF8
    }
    Copy-Item -Path (Join-Path $root "LICENSE") -Destination (Join-Path $stage "LICENSE.txt") -Force

    $zipPath = Join-Path $root "dist\OCRTrans-$version-win64-cpu.zip"
    if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
    Write-Host ">> zipping -> $zipPath"
    Compress-Archive -Path $stage -DestinationPath $zipPath -CompressionLevel Optimal
    $zipMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 0)
    $hash = (Get-FileHash $zipPath -Algorithm SHA256).Hash
    "$hash  OCRTrans-$version-win64-cpu.zip" | Set-Content "$zipPath.sha256" -Encoding ascii
    Write-Host ">> zip: $zipPath ($zipMb MB)" -ForegroundColor Green
    Write-Host ">> sha256: $hash"
}
