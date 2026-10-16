param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot
Push-Location $projectRoot
try {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python venv creation failed. Pass -Python with a Python 3.12 executable." }
    & ".venv/Scripts/python.exe" -m pip install -r backend/requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed" }
    Push-Location frontend
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
    } finally { Pop-Location }
} finally { Pop-Location }

