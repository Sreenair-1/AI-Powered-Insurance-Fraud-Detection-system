#!/usr/bin/env bash
# Start the FastAPI backend (port 8000) and the Vite web app (port 5173) together.
# Ctrl+C stops both. Usage: ./scripts/dev.sh   (or: make dev)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python3; fi

[ -d web/node_modules ] || (cd web && npm install)

"$PY" -m uvicorn api.main:app --reload --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

cd web && npm run dev
