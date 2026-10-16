param(
    [int]$Port = 8000,
    [string]$Python = '',
    [ValidateSet('local','offline','ollama')][string]$Profile = 'local',
    [ValidateSet('kg','ddxplus','recall773','random')][string]$Reasoning = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot
if (!$Python) {
    if ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX 'python.exe'))) {
        $Python = Join-Path $env:CONDA_PREFIX 'python.exe'
    } elseif (Test-Path (Join-Path $projectRoot '.venv/Scripts/python.exe')) {
        $Python = Join-Path $projectRoot '.venv/Scripts/python.exe'
    } else {
        $command = Get-Command python -ErrorAction SilentlyContinue
        if (!$command) { throw 'Activate your existing Conda environment, or pass -Python with its python.exe path.' }
        $Python = $command.Source
    }
}
if (!(Test-Path (Join-Path $projectRoot 'frontend/dist/index.html'))) {
    throw 'Build the frontend first: cd frontend; npm run build'
}
if ($Profile -eq 'ollama') {
    # Local Ollama model instead of the bundled Qwen3-4B. Ollama must be running.
    $env:AI_PROFILE = 'local'
    $env:LLM_BACKEND = 'ollama'
    if (!$env:OLLAMA_MODEL) { $env:OLLAMA_MODEL = 'qwen3:8b' }
} else {
    $env:AI_PROFILE = $Profile
    $env:LLM_BACKEND = 'transformers'
}
$env:DIALOGUE_ENGINE = 'agent'
$env:NLP_ENGINE = 'rules'
$env:NLP_USE_TRANSFORMERS = 'false'
if (!$env:AGENT_REPLY_PIPELINE) { $env:AGENT_REPLY_PIPELINE = 'direct' }
# kg = PrimeKG recall + informed search + drug relations (backend/app/services/kg_reasoning_service.py).
# ddxplus = teammate's DDXPlus recall model + informed search + PrimeKG drugs (-Reasoning ddxplus).
# recall773 = teammate's full 773-disease tri-state model + informed search + PrimeKG drugs (-Reasoning recall773).
if ($Reasoning) { $env:REASONING_MODE = $Reasoning }
# Set $env:REASONING_MODE='random' before running to use the old seeded test adapters.
if (!$env:REASONING_MODE) { $env:REASONING_MODE = 'kg' }
if (!$env:KG_NODE_FILE) { $env:KG_NODE_FILE = Join-Path $projectRoot 'backend/app/data/kg_nodes.csv' }
if ($Profile -eq 'local') {
    if (!$env:LOCAL_MODEL_PATH) { $env:LOCAL_MODEL_PATH = Join-Path $projectRoot 'models/Qwen3-4B-Instruct-2507' }
    if (!$env:EMBEDDING_MODEL_PATH) { $env:EMBEDDING_MODEL_PATH = Join-Path $projectRoot 'models/all-MiniLM-L6-v2' }
    foreach ($modelPath in @($env:LOCAL_MODEL_PATH, $env:EMBEDDING_MODEL_PATH)) {
        if (!(Test-Path -LiteralPath $modelPath -PathType Container)) { throw "Local model directory missing: $modelPath" }
    }
    if (!$env:LOCAL_MODEL_4BIT) { $env:LOCAL_MODEL_4BIT = 'true' }
    $env:HF_HUB_OFFLINE = '1'
}
Push-Location $projectRoot
try {
    Write-Host "Python: $Python"
    Write-Host "Profile: $Profile | Dialogue: agent | Adapters: $env:REASONING_MODE"
    if ($Profile -eq 'ollama') { Write-Host "Ollama model: $env:OLLAMA_MODEL | Embeddings: $(if ($env:OLLAMA_EMBED_MODEL) { $env:OLLAMA_EMBED_MODEL } else { 'none (BM25 retrieval)' })" }
    Write-Host "Logs: this terminal + backend/logs/chatbot.log (LOG_LEVEL=$(if ($env:LOG_LEVEL) { $env:LOG_LEVEL } else { 'INFO' }); set `$env:LOG_LEVEL='DEBUG' for full search tables)"
    Write-Host "Open http://127.0.0.1:$Port/ . Keep this terminal open; Ctrl+C stops the server."
    & $Python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port $Port --no-access-log
    if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE. Check the selected Python environment and port." }
} finally { Pop-Location }

