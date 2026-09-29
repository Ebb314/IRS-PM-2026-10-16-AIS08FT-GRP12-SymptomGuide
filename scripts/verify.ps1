param([switch]$Browser)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot
Push-Location $projectRoot
try {
    & ".venv/Scripts/python.exe" -m pytest backend/tests -q
    if ($LASTEXITCODE -ne 0) { throw "Backend tests failed" }
    & ".venv/Scripts/python.exe" scripts/evaluate.py --check
    if ($LASTEXITCODE -ne 0) { throw "NLP evaluation failed" }
    Push-Location frontend
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
        if ($Browser) {
            & node browser-smoke.cjs
            if ($LASTEXITCODE -ne 0) { throw "Browser test failed. Start scripts/run.ps1 in a separate terminal first." }
        }
    } finally { Pop-Location }
} finally { Pop-Location }

