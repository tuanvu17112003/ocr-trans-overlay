# Install OCRTrans into .venv. Default is the CUDA build; -Cpu downloads the
# CPU only wheels (~400 MB instead of ~3 GB) and uses the Argos engine.
#   .\install.ps1          GPU install (NLLB + Argos)
#   .\install.ps1 -Cpu     CPU install (Argos + online engines)
#   .\install.ps1 -DryRun  print the commands, change nothing
param(
    [switch]$Cpu,
    [switch]$DryRun
)

$requirementFile = if ($Cpu) { "requirements-cpu.txt" } else { "requirements.txt" }

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python launcher 'py' not found. Install Python 3.13 from python.org or the Microsoft Store, then run this again."
    exit 1
}
& py -3.13 --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python 3.13 is required (this project targets 3.13). Install it, then run this again."
    exit 1
}

$steps = @(
    @("py", "-3.13", "-m", "venv", ".venv"),
    @(".venv\Scripts\python.exe", "-m", "pip", "install", "--upgrade", "pip"),
    @(".venv\Scripts\python.exe", "-m", "pip", "install", "-r", $requirementFile),
    @(".venv\Scripts\python.exe", "-m", "pip", "install", "-e", ".", "--no-deps")
)

foreach ($step in $steps) {
    $line = $step -join " "
    if ($DryRun) {
        Write-Host $line
        continue
    }
    Write-Host ">> $line"
    & $step[0] $step[1..($step.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: command failed: $line"
        exit $LASTEXITCODE
    }
}

if ($DryRun) {
    exit 0
}

Write-Host ""
if ($Cpu) {
    Write-Host "Installed the CPU build. The app falls back to the 'argos' engine; NLLB needs the GPU build."
} else {
    Write-Host "Installed the CUDA build."
}
Write-Host "The first Start downloads the OCR/translation weights (nothing else is downloaded before that)."
Write-Host "Run it with: run.bat"
