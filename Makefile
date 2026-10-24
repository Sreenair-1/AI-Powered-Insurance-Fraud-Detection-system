# Convenience targets. On Windows without make, use scripts/dev.ps1 instead.
PY ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python3))

.PHONY: dev api web test lint build benchmarks

dev:            ## Start API (:8000) and web app (:5173) together
	./scripts/dev.sh

api:            ## Start only the FastAPI backend
	$(PY) -m uvicorn api.main:app --reload --port 8000

web:            ## Start only the Vite dev server
	cd web && npm run dev

test:           ## Python test suite
	$(PY) -m pytest tests -q

benchmarks:     ## Tampering + fraud-ring benchmarks -> models/benchmarks/
	$(PY) scripts/run_benchmarks.py

RUFF_PATHS = api tests/test_api.py src/graph/rings.py src/graph/ring_benchmark.py src/forensics/benchmark.py \
    scripts/run_benchmarks.py tests/test_rings.py tests/test_tampering_benchmark.py

lint:           ## ruff (api + benchmarks) + eslint + tsc (web)
	$(PY) -m ruff check $(RUFF_PATHS)
	cd web && npm run lint && npm run typecheck

build:          ## Production web build
	cd web && npm run build
