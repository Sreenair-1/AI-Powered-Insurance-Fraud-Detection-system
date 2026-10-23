# Start the FastAPI backend (port 8000) and the Vite web app (port 5173) together.
# Closing this window / Ctrl+C stops the web app; the API process is stopped on exit.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/dev.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

if (-not (Test-Path (Join-Path $root "web\node_modules"))) {
    Push-Location web; npm install; Pop-Location
}

$api = Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000" -PassThru -NoNewWindow
try {
    Push-Location web
    npm run dev
} finally {
    Pop-Location
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
}
