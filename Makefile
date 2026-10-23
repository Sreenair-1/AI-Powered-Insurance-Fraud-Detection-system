# Convenience targets. On Windows without make, use scripts/dev.ps1 instead.
PY ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python3))

.PHONY: dev api web test lint build

dev:            ## Start API (:8000) and web app (:5173) together
	./scripts/dev.sh

api:            ## Start only the FastAPI backend
	$(PY) -m uvicorn api.main:app --reload --port 8000

web:            ## Start only the Vite dev server
	cd web && npm run dev

test:           ## Python test suite
	$(PY) -m pytest tests -q

lint:           ## ruff (api) + eslint + tsc (web)
	$(PY) -m ruff check api tests/test_api.py
	cd web && npm run lint && npm run typecheck

build:          ## Production web build
	cd web && npm run build
