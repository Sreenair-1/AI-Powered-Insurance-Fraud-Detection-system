# Running, testing and configuration

## Requirements

- Python 3.12 with `requirements.txt` (pandas, numpy, scikit-learn, xgboost,
  shap, networkx, opencv-python, Pillow, streamlit, matplotlib, pypdf,
  pypdfium2, fastapi, uvicorn, python-multipart). `requirements-dev.txt` adds
  pytest, httpx and ruff.
- Node 20+ for the web app in `web/`.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

## Raw data layout

`data/` is gitignored. Every loader in `src/data/loaders.py` raises
`DatasetFileMissingError` naming the exact path when a file is missing.
Paths are defined in `src/data/schemas.py`:

| Path | Source |
| --- | --- |
| `data/raw/motor/fraud_oracle.csv` | Kaggle `shivamb/vehicle-claim-fraud-detection` |
| `data/raw/health/Train-1542865627584.csv` | Kaggle `rohitrox/healthcare-provider-fraud-detection-analysis` (provider labels) |
| `data/raw/health/Train_Beneficiarydata-1542865627584.csv` | same, beneficiaries |
| `data/raw/health/Train_Inpatientdata-1542865627584.csv` | same, inpatient claims |
| `data/raw/health/Train_Outpatientdata-1542865627584.csv` | same, outpatient claims |
| `data/raw/demographics/Indian_Insurance_Data.csv` | Kaggle `patelparth3399/indian-insurance-premium-prediction-dataset` |

The health `Test-*` files sit next to the train files but carry no labels.

## Commands

| Task | Command |
| --- | --- |
| Train both models | `./.venv/Scripts/python.exe scripts/train_models.py --domain both` |
| Component benchmarks | `./.venv/Scripts/python.exe scripts/run_benchmarks.py` (or `make benchmarks`) |
| API + web app | `make dev`, or `powershell -ExecutionPolicy Bypass -File scripts/dev.ps1` on Windows |
| API only | `./.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000` |
| Streamlit app | `./.venv/Scripts/python.exe -m streamlit run app/main.py` |
| Python tests | `./.venv/Scripts/python.exe -m pytest tests/ -q` |
| Lint | `make lint` (ruff on the API and benchmark modules, eslint + tsc on `web/`) |
| Web checks | `cd web && npm run lint && npm run typecheck && npm run check:contrast && npm run build` |

`scripts/run_benchmarks.py` needs no raw data. `--component tampering|rings`
runs one benchmark, `--images-dir <folder>` tampers your own bill images
instead of rendered ones, and `--n-bills`, `--n-claims` and `--ring-seeds`
size the runs.

## Tests

Most tests use small in-memory fixtures. In CI, `tests/fixtures/setup_test_data.py`
writes tiny mock CSVs to the real `data/raw/` paths for the tests that read
files. **Do not run it locally**: it overwrites the Kaggle files with mock data.
The API tests (`tests/test_api.py`) fit tiny models into a temporary directory
and point every path setting below at temporary locations, so they never touch
`models/` or the review log.

| File | Covers |
| --- | --- |
| `tests/test_api.py` | Every FastAPI endpoint, including reviews, audit reports and the workload curve |
| `tests/test_rings.py` | Entity resolution, the ring detector and the ring benchmark |
| `tests/test_tampering_benchmark.py` | Bill rendering, edits, metadata handling and the tampering report |
| `tests/test_ela.py`, `test_copy_move.py`, `test_metadata.py`, `test_forensics_pdf.py`, `test_tampering_evaluation.py` | Document forensics |
| `tests/test_graph.py`, `test_feature_*.py` | Graph construction and feature fusion |
| `tests/test_model_*.py`, `test_threshold_selection.py`, `test_shap_*.py` | Models, evaluation, persistence, SHAP |
| `tests/test_app_*.py` | Streamlit app helpers |

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRAUD_APP_MOTOR_MODEL_DIR` | `models/motor` | Motor artifact the apps load |
| `FRAUD_APP_HEALTH_MODEL_DIR` | `models/health` | Health artifact the apps load |
| `FRAUD_APP_BENCHMARKS_DIR` | `models/benchmarks` | Benchmark JSON the API serves |
| `FRAUD_APP_REVIEWS_PATH` | `data/reviews/reviews.jsonl` | Append-only investigator review log |
| `INSURESIGHT_API_URL` | `http://127.0.0.1:8000` | API the Vite dev server proxies `/api` to |
| `INSURESIGHT_CORS_ORIGINS` | – | Extra browser origins the API accepts |
| `VITE_API_BASE` | `/api` | API base for a deployed web build |
