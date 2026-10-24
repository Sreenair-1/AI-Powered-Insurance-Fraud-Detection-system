"""FastAPI application: `uvicorn api.main:app --reload`.

Models are loaded once in the lifespan hook (never trained in-process);
sampled datasets are built lazily per sample-size combination and cached in
`api.state.AppState`.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from api import schemas, services
from api.audit_html import render_audit_html
from api.state import DOMAINS, AppState, DataUnavailable, ModelUnavailable, SampleParams
from app import config
from app.claim_view import ClaimViewError
from app.data_loader import DatasetNotAvailableError

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf")
CORS_ORIGINS_ENV = "INSURESIGHT_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "insuresight"):
        app.state.insuresight = AppState.load()
    yield


app = FastAPI(
    title="InsureSight AI API",
    version="1.0.0",
    description=(
        "Decision-support API for health and motor claim risk triage. Scores are model "
        "probabilities for prioritising human review - they are not determinations."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get(CORS_ORIGINS_ENV, DEFAULT_CORS_ORIGINS).split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@app.exception_handler(ModelUnavailable)
async def _model_unavailable(_: Request, exc: ModelUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc), "code": "model_unavailable"})


@app.exception_handler(DataUnavailable)
async def _data_unavailable(_: Request, exc: DataUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc), "code": "data_unavailable"})


@app.exception_handler(ClaimViewError)
async def _claim_not_found(_: Request, exc: ClaimViewError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc), "code": "claim_not_found"})


@app.exception_handler(DatasetNotAvailableError)
async def _claim_row_missing(_: Request, exc: DatasetNotAvailableError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc), "code": "claim_not_found"})


@app.exception_handler(services.DocumentError)
async def _bad_document(_: Request, exc: services.DocumentError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc), "code": "invalid_document"})


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_state(request: Request) -> AppState:
    return request.app.state.insuresight


def valid_domain(domain: str) -> str:
    if domain not in DOMAINS:
        raise HTTPException(status_code=404, detail=f"Unknown domain {domain!r}; expected one of {list(DOMAINS)}")
    return domain


def sample_params(
    domain: Annotated[str, Depends(valid_domain)],
    sample_size: Annotated[
        int, Query(ge=100, le=config.MAX_MOTOR_SAMPLE_SIZE, description="Motor claims sampled")
    ] = config.DEFAULT_MOTOR_SAMPLE_SIZE,
    inpatient_sample_size: Annotated[
        int, Query(ge=20, le=config.MAX_HEALTH_INPATIENT_SAMPLE_SIZE, description="Health inpatient claims sampled")
    ] = config.DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE,
    outpatient_sample_size: Annotated[
        int, Query(ge=40, le=config.MAX_HEALTH_OUTPATIENT_SAMPLE_SIZE, description="Health outpatient claims sampled")
    ] = config.DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE,
) -> SampleParams:
    return SampleParams(domain, sample_size, inpatient_sample_size, outpatient_sample_size)


StateDep = Annotated[AppState, Depends(get_state)]
DomainDep = Annotated[str, Depends(valid_domain)]
ParamsDep = Annotated[SampleParams, Depends(sample_params)]


async def _read_upload(upload: UploadFile) -> tuple[bytes, str]:
    filename = upload.filename or "document"
    if not filename.lower().endswith(ALLOWED_UPLOAD_SUFFIXES):
        raise HTTPException(
            status_code=422, detail=f"Unsupported file type; allowed: {', '.join(ALLOWED_UPLOAD_SUFFIXES)}"
        )
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than the 20 MB upload limit")
    return data, filename


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=schemas.HealthResponse, tags=["system"])
def health(state: StateDep) -> dict[str, Any]:
    domains = []
    for domain in DOMAINS:
        slot = state.models[domain]
        entry: dict[str, Any] = {
            "domain": domain,
            "model_loaded": slot.model is not None,
            "model_dir": str(slot.directory),
            "error": slot.error,
        }
        if slot.model is not None:
            entry["n_features"] = len(services.model_feature_names(slot.model))
            entry["uses_graph_features"] = services.uses_graph_features(slot.model)
        domains.append(entry)
    return {
        "status": "ok" if all(d["model_loaded"] for d in domains) else "degraded",
        "domains": domains,
        "risk_bands": services.risk_bands(),
    }


@app.get("/domains/{domain}/overview", response_model=schemas.OverviewResponse, tags=["claims"])
def overview(state: StateDep, params: ParamsDep) -> dict[str, Any]:
    return services.overview(state, state.sample(params))


@app.get("/domains/{domain}/claims", response_model=schemas.ClaimsPage, tags=["claims"])
def claims(
    state: StateDep,
    params: ParamsDep,
    search: str = "",
    risk: Annotated[Optional[list[Literal["low", "medium", "high"]]], Query()] = None,
    min_probability: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    segment: Annotated[Optional[list[str]], Query()] = None,
    sort: Literal["fraud_probability", "claim_key", "risk_category"] = "fraud_probability",
    order: Literal["asc", "desc"] = "desc",
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=5, le=100)] = 25,
) -> dict[str, Any]:
    return services.claims_page(
        state.sample(params), search, list(risk or []), min_probability, segment, sort, order, page, page_size
    )


@app.get("/domains/{domain}/claims/{claim_key}", response_model=schemas.ClaimDetail, tags=["claims"])
def claim_detail(state: StateDep, params: ParamsDep, claim_key: str) -> dict[str, Any]:
    return services.claim_detail(state, state.sample(params), claim_key)


@app.get("/domains/{domain}/claims/{claim_key}/graph", response_model=schemas.GraphResponse, tags=["claims"])
def claim_graph(state: StateDep, params: ParamsDep, claim_key: str) -> dict[str, Any]:
    return services.claim_graph(state.sample(params), claim_key)


@app.post(
    "/domains/{domain}/claims/{claim_key}/document",
    response_model=schemas.DocumentRescoreResponse,
    tags=["forensics"],
)
async def claim_document(
    state: StateDep,
    params: ParamsDep,
    claim_key: str,
    file: Annotated[UploadFile, File(description="Claim document: PNG, JPEG, TIFF, BMP or PDF")],
    page_index: Annotated[int, Form(ge=0)] = 0,
) -> dict[str, Any]:
    data, filename = await _read_upload(file)
    # Dataset build, forensics and SHAP are CPU-bound and synchronous: run them
    # off the event loop so other requests are not blocked meanwhile.
    sample = await run_in_threadpool(state.sample, params)
    return await run_in_threadpool(
        services.rescore_claim_with_document, state, sample, claim_key, data, filename, page_index
    )


@app.get("/domains/{domain}/claims/{claim_key}/review", response_model=schemas.ClaimReview, tags=["review"])
def claim_review(state: StateDep, params: ParamsDep, claim_key: str) -> dict[str, Any]:
    services.require_claim(state.sample(params), claim_key)
    return services.claim_review(state, params.domain, claim_key)


@app.post("/domains/{domain}/claims/{claim_key}/review", response_model=schemas.ClaimReview, tags=["review"])
def record_review(state: StateDep, params: ParamsDep, claim_key: str, body: schemas.ReviewInput) -> dict[str, Any]:
    return services.record_review(state, state.sample(params), claim_key, body.outcome, body.note, body.reviewer)


@app.get("/domains/{domain}/reviews", response_model=schemas.ReviewSummary, tags=["review"])
def review_summary(state: StateDep, domain: DomainDep) -> dict[str, Any]:
    return services.review_summary(state, domain)


def _audit_response(report: dict[str, Any], fmt: str) -> Response:
    name = f"audit-{report['domain']}-{report['claim_key'].replace(':', '_')}"
    if fmt == "html":
        return HTMLResponse(render_audit_html(report))
    return JSONResponse(report, headers={"Content-Disposition": f'attachment; filename="{name}.json"'})


@app.get("/domains/{domain}/claims/{claim_key}/audit-report", tags=["review"])
async def audit_report(
    state: StateDep,
    params: ParamsDep,
    claim_key: str,
    format: Annotated[Literal["json", "html"], Query(description="json download or printable html")] = "json",
) -> Response:
    sample = await run_in_threadpool(state.sample, params)
    report = await run_in_threadpool(services.audit_report, state, sample, claim_key)
    return _audit_response(report, format)


@app.post("/domains/{domain}/claims/{claim_key}/audit-report", tags=["review"])
async def audit_report_with_document(
    state: StateDep,
    params: ParamsDep,
    claim_key: str,
    file: Annotated[UploadFile, File(description="Document whose forensic findings go into the report")],
    page_index: Annotated[int, Form(ge=0)] = 0,
    format: Annotated[Literal["json", "html"], Query(description="json download or printable html")] = "json",
) -> Response:
    data, filename = await _read_upload(file)
    sample = await run_in_threadpool(state.sample, params)
    report = await run_in_threadpool(services.audit_report, state, sample, claim_key, (data, filename, page_index))
    return _audit_response(report, format)


@app.get("/domains/{domain}/schema", response_model=schemas.ClaimSchema, tags=["assessment"])
def claim_schema(state: StateDep, domain: DomainDep) -> dict[str, Any]:
    return services.claim_schema(state, domain)


@app.post("/domains/{domain}/assess", response_model=schemas.AssessResponse, tags=["assessment"])
async def assess(
    state: StateDep,
    domain: DomainDep,
    attributes: Annotated[str, Form(description="JSON object of claim attributes (see /schema)")] = "{}",
    file: Annotated[Optional[UploadFile], File(description="Optional supporting document")] = None,
    page_index: Annotated[int, Form(ge=0)] = 0,
) -> dict[str, Any]:
    try:
        parsed = json.loads(attributes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"attributes is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="attributes must be a JSON object")
    document = None
    if file is not None and file.filename:
        data, filename = await _read_upload(file)
        document = (data, filename, page_index)
    try:
        return await run_in_threadpool(services.assess_new_claim, state, domain, parsed, document)
    except ValueError as exc:
        if isinstance(exc, (services.DocumentError, ClaimViewError)):
            raise
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/domains/{domain}/insights", response_model=schemas.InsightsResponse, tags=["insights"])
def insights(state: StateDep, domain: DomainDep) -> dict[str, Any]:
    return services.insights(state, domain)
