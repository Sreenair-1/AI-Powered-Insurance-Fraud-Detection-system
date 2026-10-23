"""Adapters from the existing pipeline to API response shapes.

Every score, SHAP value, forensic result and metric here comes from an
existing function in `app/` or `src/`, or is read from a `models/` sidecar.
This module only selects, groups, filters, paginates and serializes.
"""
from __future__ import annotations

import io
import json
import math
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

from api.serialization import image_to_data_url, to_jsonable
from api.state import HEALTH_DOMAIN, MOTOR_DOMAIN, AppState, Model, ScoredSample
from app import data_loader, model_loader
from app.claim_view import (
    GRAPH_SIGNAL_COLUMNS,
    ClaimViewError,
    ForensicUploadResult,
    assess_claim,
    assess_new_claim_document,
    build_new_claim_features,
    claim_row,
    generate_new_claim_id,
    rescore_with_forensics,
    run_forensic_checks,
)
from app.ui_components import extract_model_feature_importances
from src.data.schemas import HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL, HEALTH_PROVIDER_LABEL_COL, MOTOR_TARGET
from src.data.synthetic import RING_ID_COL, RING_LABEL_COL
from src.features.structured import CLAIM_KEY_COL
from src.forensics import (
    PDFLoadError,
    evaluate_document_tampering,
    extract_pdf_text,
    get_pdf_info,
    render_pdf_page,
)
from src.graph import (
    BANK_ACCOUNT_NODE_TYPE,
    CLAIM_NODE_TYPE,
    EDGE_TYPE_ATTR,
    GARAGE_NODE_TYPE,
    NODE_TYPE_ATTR,
)
from src.pipeline.inference import (
    DEFAULT_RISK_THRESHOLDS,
    ClaimRiskAssessment,
)

RISK_ORDER = {"high": 0, "medium": 1, "low": 2}

# Graph-derived columns `src.features.fusion` adds to the fused matrix (see
# docs/graph.md). Used only to report whether a saved model consumes them.
GRAPH_FEATURE_COLUMNS = frozenset(
    {
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
        "community_id",
        "num_entities",
        "max_entity_claim_degree",
        "mean_entity_claim_degree",
        "has_shared_entity",
    }
)

# Same attribute groupings as the Streamlit claim detail (app/main.py).
MOTOR_GROUPS = {
    "Policy": ("PolicyType", "BasePolicy", "Deductible", "Days_Policy_Accident", "Days_Policy_Claim"),
    "Vehicle": ("Make", "VehicleCategory", "VehiclePrice", "AgeOfVehicle", "NumberOfCars"),
    "Incident": ("AccidentArea", "Fault", "PoliceReportFiled", "WitnessPresent", "Month", "Year"),
    "Driver": ("Age", "Sex", "MaritalStatus", "DriverRating", "AgeOfPolicyHolder"),
    "Linked entities (synthetic)": ("GarageID", "BankAccountID"),
}
HEALTH_GROUPS = {
    "Claim": ("ClaimID", "claim_type", "ClaimStartDt", "ClaimEndDt", "InscClaimAmtReimbursed", "DeductibleAmtPaid"),
    "Provider": ("Provider", "AttendingPhysician", "OperatingPhysician", "OtherPhysician"),
    "Beneficiary": ("BeneID", "num_diagnosis_codes", "num_procedure_codes"),
}

# Target / label columns never shown to investigators: the UI is decision
# support for unlabelled claims, so ground-truth labels must not leak into it.
HIDDEN_ATTRIBUTE_COLUMNS = frozenset(
    {MOTOR_TARGET, RING_LABEL_COL, RING_ID_COL, HEALTH_PROVIDER_LABEL_COL, HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL}
)

SEGMENT_FIELD = {MOTOR_DOMAIN: "AccidentArea", HEALTH_DOMAIN: "claim_type"}
SYNTHETIC_NODE_TYPES = {MOTOR_DOMAIN: [GARAGE_NODE_TYPE, BANK_ACCOUNT_NODE_TYPE], HEALTH_DOMAIN: []}

MAX_GRAPH_NODES = 80
DOCUMENT_FACTOR_PREFIX = "Document Forensics:"
FUSION_NOTE = (
    "After a document upload the model score is combined with the document tampering score by a "
    "rule-based adjustment (app.claim_view.fuse_assessment_with_document_evidence): "
    "p' = min(0.99, p + (1 - p) x tamper x 0.85). Factors marked 'document_heuristic' are fixed "
    "rule weights, not SHAP values."
)
LIMITATIONS = {
    HEALTH_DOMAIN: [
        "The health label is a provider-level proxy: each claim inherits its provider's PotentialFraud "
        "flag. Metrics measure separation of that proxy, not verified claim-level outcomes.",
        "Document forensics only applies to files uploaded in this session; the bulk dataset has no documents.",
    ],
    MOTOR_DOMAIN: [
        "Motor garage and bank-account links are synthetic (seeded augmentation). They are shown as "
        "investigative context and are not observed insurance data.",
        "Document forensics only applies to files uploaded in this session; the bulk dataset has no documents.",
    ],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def risk_bands() -> dict[str, float]:
    return {"medium": DEFAULT_RISK_THRESHOLDS.medium, "high": DEFAULT_RISK_THRESHOLDS.high}


def model_feature_names(model: Model) -> list[str]:
    names = getattr(model, "feature_columns", None)
    if names is None:
        names = getattr(model, "feature_names_", None)
    return list(names or [])


def uses_graph_features(model: Model) -> bool:
    return any(f in GRAPH_FEATURE_COLUMNS for f in model_feature_names(model))


def assessment_to_dict(assessment: ClaimRiskAssessment) -> dict[str, Any]:
    summary = assessment.to_ui_summary()
    for factor in summary["top_factors"]:
        factor["source"] = (
            "document_heuristic" if str(factor["feature"]).startswith(DOCUMENT_FACTOR_PREFIX) else "model_shap"
        )
    return to_jsonable(summary)


def _summary_row(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    return {
        "claim_key": str(row[CLAIM_KEY_COL]),
        "fraud_probability": float(row["fraud_probability"]),
        "risk_category": str(row["risk_category"]),
        "attributes": {c: to_jsonable(row[c]) for c in columns},
    }


def _attribute_columns(table: pd.DataFrame) -> list[str]:
    fixed = {CLAIM_KEY_COL, "domain", "fraud_probability", "risk_category"}
    return [c for c in table.columns if c not in fixed]


def _priority_sorted(table: pd.DataFrame) -> pd.DataFrame:
    ordered = table.assign(_order=table["risk_category"].map(RISK_ORDER))
    return ordered.sort_values(["_order", "fraud_probability"], ascending=[True, False], kind="stable").drop(
        columns="_order"
    )


# ---------------------------------------------------------------------------
# Overview and claims table
# ---------------------------------------------------------------------------


def overview(state: AppState, sample: ScoredSample, top_n: int = 10) -> dict[str, Any]:
    domain = sample.params.domain
    table = sample.table
    n = len(table)
    counts = {k: int((table["risk_category"] == k).sum()) if n else 0 for k in ("high", "medium", "low")}
    review = counts["high"] + counts["medium"]

    edges = np.linspace(0.0, 1.0, 11)
    hist_counts, _ = np.histogram(table["fraud_probability"].to_numpy(dtype=float), bins=edges) if n else (
        np.zeros(10, dtype=int),
        None,
    )
    histogram = [
        {"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(hist_counts[i])} for i in range(10)
    ]

    seg_field = SEGMENT_FIELD[domain]
    segments = []
    if n and seg_field in table.columns:
        grouped = table.groupby(table[seg_field].astype(str))["risk_category"].value_counts().unstack(fill_value=0)
        for seg, row in grouped.iterrows():
            segments.append({"segment": str(seg), **{k: int(row.get(k, 0)) for k in ("low", "medium", "high")}})

    columns = _attribute_columns(table)
    top = _priority_sorted(table).head(top_n) if n else table
    return {
        "domain": domain,
        "sample": sample.params.describe(),
        "total_claims": n,
        "risk_counts": counts,
        "review_count": review,
        "review_rate": (review / n) if n else None,
        "risk_bands": risk_bands(),
        "probability_histogram": histogram,
        "segment_field": seg_field if segments else None,
        "segments": segments,
        "global_importance": extract_model_feature_importances(state.model(domain), top_n=8),
        "top_claims": [_summary_row(r, columns) for _, r in top.iterrows()],
    }


SORTABLE = {"fraud_probability", "claim_key", "risk_category"}


def claims_page(
    sample: ScoredSample,
    search: str = "",
    risk: Optional[list[str]] = None,
    min_probability: float = 0.0,
    segment: Optional[list[str]] = None,
    sort: str = "fraud_probability",
    order: str = "desc",
    page: int = 1,
    page_size: int = 25,
) -> dict[str, Any]:
    domain = sample.params.domain
    table = sample.table
    seg_field = SEGMENT_FIELD[domain]
    seg_options = sorted(table[seg_field].dropna().astype(str).unique().tolist()) if seg_field in table.columns else []

    filtered = table
    if risk:
        filtered = filtered[filtered["risk_category"].isin(risk)]
    if min_probability > 0:
        filtered = filtered[filtered["fraud_probability"] >= min_probability]
    if segment and seg_field in filtered.columns:
        filtered = filtered[filtered[seg_field].astype(str).isin(segment)]
    needle = search.strip().lower()
    if needle:
        haystack = filtered[CLAIM_KEY_COL].astype(str).str.lower()
        for col in ("Provider", "BeneID", "Make", "AttendingPhysician"):
            if col in filtered.columns:
                haystack = haystack + " " + filtered[col].astype(str).str.lower()
        filtered = filtered[haystack.str.contains(needle, regex=False)]

    ascending = order == "asc"
    if sort == "risk_category":
        filtered = filtered.assign(_order=filtered["risk_category"].map(RISK_ORDER)).sort_values(
            ["_order", "fraud_probability"], ascending=[not ascending, ascending], kind="stable"
        ).drop(columns="_order")
    else:
        filtered = filtered.sort_values(sort, ascending=ascending, kind="stable")

    total = len(filtered)
    pages = max(1, math.ceil(total / page_size))
    page = min(page, pages)
    start = (page - 1) * page_size
    columns = _attribute_columns(table)
    return {
        "domain": domain,
        "sample": sample.params.describe(),
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
        "columns": columns,
        "segment_field": seg_field if seg_options else None,
        "segment_options": seg_options,
        "items": [_summary_row(r, columns) for _, r in filtered.iloc[start : start + page_size].iterrows()],
    }


# ---------------------------------------------------------------------------
# Claim detail and graph
# ---------------------------------------------------------------------------


def _require_claim(sample: ScoredSample, claim_key: str) -> None:
    if claim_key not in set(sample.dataset.X[CLAIM_KEY_COL].astype(str)):
        raise ClaimViewError(f"claim_key {claim_key!r} not found in the loaded sample")


def _raw_claim(sample: ScoredSample, claim_key: str) -> pd.Series:
    if sample.params.domain == MOTOR_DOMAIN:
        return data_loader.motor_claim_row(sample.dataset, claim_key)
    return data_loader.health_claim_row(sample.dataset, claim_key)


def claim_detail(state: AppState, sample: ScoredSample, claim_key: str) -> dict[str, Any]:
    _require_claim(sample, claim_key)
    domain = sample.params.domain
    model = state.model(domain)
    assessment = assess_claim(domain, model, claim_row(sample.dataset, claim_key), DEFAULT_RISK_THRESHOLDS)

    raw = _raw_claim(sample, claim_key)
    attributes = {str(k): to_jsonable(v) for k, v in raw.items() if k not in HIDDEN_ATTRIBUTE_COLUMNS}
    groups_def = MOTOR_GROUPS if domain == MOTOR_DOMAIN else HEALTH_GROUPS
    groups = []
    grouped: set[str] = set()
    for name, fields in groups_def.items():
        present = [f for f in fields if f in attributes]
        grouped.update(present)
        if present:
            groups.append({"name": name, "fields": present})
    other = [k for k, v in attributes.items() if k not in grouped and v not in (None, "")]
    if other:
        groups.append({"name": "Other recorded fields", "fields": other})

    fused = sample.dataset.X[sample.dataset.X[CLAIM_KEY_COL] == claim_key].iloc[0]
    signal_cols = [c for c in (*GRAPH_SIGNAL_COLUMNS, "num_entities", "betweenness_centrality") if c in fused.index]
    return {
        "domain": domain,
        "claim_key": claim_key,
        "assessment": assessment_to_dict(assessment),
        "attributes": attributes,
        "attribute_groups": groups,
        "graph_signals": {c: to_jsonable(fused[c]) for c in signal_cols},
        "model_uses_graph_features": uses_graph_features(model),
        "risk_bands": risk_bands(),
    }


def _node_label(node: str) -> str:
    return node.split(":", 1)[-1]


def claim_graph(sample: ScoredSample, claim_key: str) -> dict[str, Any]:
    """Ego network: the claim, its entities, and other claims sharing them.

    Bounded to `MAX_GRAPH_NODES` the same way the Streamlit view is.
    """
    _require_claim(sample, claim_key)
    graph: nx.Graph = sample.dataset.graph
    if claim_key not in graph:
        raise ClaimViewError(f"claim_key {claim_key!r} is not present in the relationship graph")

    neighborhood = {claim_key}
    for entity in graph.neighbors(claim_key):
        neighborhood.add(entity)
        neighborhood.update(n for n in graph.neighbors(entity) if n != claim_key)
    entities = sorted(n for n in neighborhood if graph.nodes[n].get(NODE_TYPE_ATTR) != CLAIM_NODE_TYPE)
    others = sorted(
        n for n in neighborhood if n != claim_key and graph.nodes[n].get(NODE_TYPE_ATTR) == CLAIM_NODE_TYPE
    )
    ordered = ([claim_key] + entities + others)[:MAX_GRAPH_NODES]
    truncated = len(neighborhood) > len(ordered)
    visible = graph.subgraph(ordered)

    scores = sample.table.set_index(CLAIM_KEY_COL)
    X = sample.dataset.X
    communities = X.set_index(CLAIM_KEY_COL)["community_id"] if "community_id" in X.columns else None

    nodes = []
    for node in ordered:
        kind = str(graph.nodes[node].get(NODE_TYPE_ATTR, "entity"))
        is_claim = kind == CLAIM_NODE_TYPE
        linked = sum(1 for n in graph.neighbors(node) if graph.nodes[n].get(NODE_TYPE_ATTR) == CLAIM_NODE_TYPE)
        entry: dict[str, Any] = {
            "id": node,
            "type": kind,
            "label": _node_label(node),
            "is_focus": node == claim_key,
            "linked_claims": 1 if is_claim else linked,
            "shared": (not is_claim) and linked > 1,
        }
        if is_claim and node in scores.index:
            entry["risk_category"] = str(scores.at[node, "risk_category"])
            entry["fraud_probability"] = float(scores.at[node, "fraud_probability"])
            if communities is not None and node in communities.index:
                entry["community_id"] = to_jsonable(communities.at[node])
        nodes.append(entry)

    edges = [
        {"source": a, "target": b, "type": str(attrs.get(EDGE_TYPE_ATTR, "related"))}
        for a, b, attrs in visible.edges(data=True)
    ]
    return {
        "domain": sample.params.domain,
        "claim_key": claim_key,
        "nodes": nodes,
        "edges": edges,
        "truncated": truncated,
        "synthetic_entities": SYNTHETIC_NODE_TYPES[sample.params.domain],
    }


# ---------------------------------------------------------------------------
# Document forensics
# ---------------------------------------------------------------------------


class DocumentError(ValueError):
    """The uploaded file could not be read or analyzed."""


def is_pdf_bytes(data: bytes, filename: str) -> bool:
    return data[:5].startswith(b"%PDF") or filename.lower().endswith(".pdf")


def _copy_move_regions(pairs: list, bin_size: float) -> list[dict[str, Any]]:
    """Group matched keypoint pairs by shared offset (as `_max_offset_cluster`
    does) and box each group with 2+ matches: a block moved as a whole."""
    clusters: dict[tuple[int, int], list] = {}
    for x1, y1, x2, y2 in pairs:
        key = (round((x2 - x1) / bin_size), round((y2 - y1) / bin_size))
        clusters.setdefault(key, []).append((x1, y1, x2, y2))
    pad = 6.0
    regions = []
    for members in sorted(clusters.values(), key=len, reverse=True):
        if len(members) < 2:
            continue
        arr = np.asarray(members, dtype=float)
        regions.append(
            {
                "source": (arr[:, 0].min() - pad, arr[:, 1].min() - pad, arr[:, 0].max() + pad, arr[:, 1].max() + pad),
                "target": (arr[:, 2].min() - pad, arr[:, 3].min() - pad, arr[:, 2].max() + pad, arr[:, 3].max() + pad),
                "matches": len(members),
            }
        )
    return regions


def _metadata_flags(forensic: ForensicUploadResult) -> list[dict[str, str]]:
    if forensic.metadata is None:
        return []
    feats = forensic.metadata.features
    flags = []
    if feats.software_editing_indicator:
        flags.append({"field": "software", "reason": "Software tag references a known editing tool"})
    if feats.datetime_original_modified_mismatch:
        flags.append({"field": "datetime_original", "reason": "Creation and modification timestamps differ"})
        flags.append({"field": "datetime_modified", "reason": "Creation and modification timestamps differ"})
    if feats.exif_malformed:
        flags.append({"field": "format", "reason": "Some metadata entries could not be parsed"})
    return flags


def analyze_document(data: bytes, filename: str, page_index: int = 0) -> tuple[ForensicUploadResult, dict[str, Any]]:
    """Run the existing forensic checks on an upload and build the report payload."""
    if not data:
        raise DocumentError("The uploaded file is empty.")
    is_pdf = is_pdf_bytes(data, filename)
    page_count = 1
    text: Optional[str] = None
    try:
        if is_pdf:
            page_count = get_pdf_info(data).page_count
            if not 0 <= page_index < page_count:
                raise DocumentError(f"page_index {page_index} is out of range for a {page_count}-page PDF.")
            # Same default render scale run_forensic_checks analyzes, so overlays line up.
            preview = render_pdf_page(data, page_index=page_index)
            text = extract_pdf_text(data, page_index=page_index)[:5000]
        else:
            if page_index != 0:
                raise DocumentError("page_index applies only to PDF uploads.")
            preview = Image.open(io.BytesIO(data))
            preview.load()
    except PDFLoadError as exc:
        raise DocumentError(f"Could not read the PDF: {exc}") from exc
    except OSError as exc:
        raise DocumentError(f"Could not read the image: {exc}") from exc

    stream = io.BytesIO(data)
    stream.name = filename
    forensic = run_forensic_checks(stream, page_index=page_index)
    if not forensic.any_available:
        raise DocumentError("None of the forensic checks could run on this file (unsupported or corrupt).")
    verdict = evaluate_document_tampering(ela=forensic.ela, copy_move=forensic.copy_move, metadata=forensic.metadata)

    pairs: list = []
    regions: list = []
    if forensic.copy_move is not None:
        pairs = list(forensic.copy_move.metadata.get("matched_pairs", []))
        regions = _copy_move_regions(pairs, float(forensic.copy_move.metadata.get("offset_bin_size", 5.0)))

    metadata_payload = None
    if forensic.metadata is not None:
        metadata_payload = {
            "structured": to_jsonable(forensic.metadata.structured_metadata),
            "features": to_jsonable(forensic.metadata.features),
            "indicators": to_jsonable(forensic.metadata.analysis.get("indicators", [])),
        }

    report = {
        "filename": filename,
        "is_pdf": is_pdf,
        "page_index": page_index,
        "page_count": page_count,
        "image_width": preview.size[0],
        "image_height": preview.size[1],
        "original_image": image_to_data_url(preview.convert("RGB"), "JPEG"),
        "ela_image": image_to_data_url(forensic.ela.ela_image) if forensic.ela else None,
        "copy_move_image": (
            image_to_data_url(forensic.copy_move.visualization, "JPEG")
            if forensic.copy_move and forensic.copy_move.visualization is not None
            else None
        ),
        "ela": to_jsonable(forensic.ela.features) if forensic.ela else None,
        "copy_move": (
            {
                "potential_duplicate_regions_found": bool(forensic.copy_move.potential_duplicate_regions_found),
                **to_jsonable(forensic.copy_move.features),
            }
            if forensic.copy_move
            else None
        ),
        "copy_move_pairs": [tuple(float(v) for v in p) for p in pairs[:400]],
        "copy_move_regions": regions,
        "metadata": metadata_payload,
        "metadata_flags": _metadata_flags(forensic),
        "extracted_text": text,
        "verdict": {
            "status": verdict.status,
            "tampering_score": float(verdict.tampering_score),
            "is_tampered": bool(verdict.is_tampered),
            "indicators": list(verdict.tampering_indicators),
            "summary": verdict.summary,
            "details": to_jsonable(verdict.details),
        },
    }
    return forensic, report


def rescore_claim_with_document(
    state: AppState, sample: ScoredSample, claim_key: str, data: bytes, filename: str, page_index: int
) -> dict[str, Any]:
    _require_claim(sample, claim_key)
    domain = sample.params.domain
    model = state.model(domain)
    forensic, report = analyze_document(data, filename, page_index)
    before = assess_claim(domain, model, claim_row(sample.dataset, claim_key), DEFAULT_RISK_THRESHOLDS)
    after = rescore_with_forensics(domain, model, sample.dataset, claim_key, forensic, DEFAULT_RISK_THRESHOLDS)
    return {
        "domain": domain,
        "claim_key": claim_key,
        "forensics": report,
        "before": assessment_to_dict(before),
        "after": assessment_to_dict(after),
        "fusion_note": FUSION_NOTE,
    }


# ---------------------------------------------------------------------------
# New-claim schema and assessment
# ---------------------------------------------------------------------------

HEALTH_CLAIM_TYPES = ["inpatient", "outpatient"]  # src.data.preprocessing.build_health_claims_table
BOOLEAN_FIELDS = {"is_deceased", "has_renal_disease"}
INTEGER_FIELDS = {
    "WeekOfMonth", "WeekOfMonthClaimed", "Age", "Deductible", "DriverRating", "Year",
    "num_diagnosis_codes", "num_procedure_codes", "chronic_condition_count",
}


def _motor_categories(model: Model) -> dict[str, list[str]]:
    dtypes = getattr(model, "_categorical_dtypes", None) or {}
    out = {}
    for col, dtype in dtypes.items():
        cats = getattr(dtype, "categories", None)
        if cats is not None:
            out[col] = [str(c) for c in cats]
    return out


def claim_schema(state: AppState, domain: str) -> dict[str, Any]:
    """Form fields = the saved model's own feature list. Defaults are whatever
    `build_new_claim_features` produces with no overrides (null = not provided)."""
    model = state.model(domain)
    features = model_feature_names(model)
    empty = ForensicUploadResult(ela=None, copy_move=None, metadata=None)
    baseline = build_new_claim_features(domain, "claim:SCHEMA", empty, {}).iloc[0]
    categories = _motor_categories(model) if domain == MOTOR_DOMAIN else {"claim_type": HEALTH_CLAIM_TYPES}

    fields = []
    for name in features:
        default = to_jsonable(baseline.get(name)) if name in baseline.index else None
        if name in categories:
            options = categories[name]
            fields.append(
                {
                    "name": name,
                    "kind": "categorical",
                    "options": options,
                    "default": str(default) if default is not None and str(default) in options else None,
                }
            )
        elif name in BOOLEAN_FIELDS:
            fields.append({"name": name, "kind": "boolean", "default": bool(default) if default is not None else False})
        else:
            kind = "integer" if name in INTEGER_FIELDS else "number"
            fields.append({"name": name, "kind": kind, "default": default, "minimum": 0})
    notes = [
        "Fields left blank are passed to the model as missing values, exactly as the Streamlit "
        "new-claim flow does; nothing is imputed by the API.",
        "A new claim has no relationship history, so it is scored as an isolated graph node.",
    ]
    return {"domain": domain, "fields": fields, "notes": notes}


def validate_attributes(state: AppState, domain: str, attributes: dict[str, Any]) -> dict[str, Any]:
    """Check user-supplied attributes against the schema; return the cleaned overrides."""
    schema = {f["name"]: f for f in claim_schema(state, domain)["fields"]}
    errors = []
    clean: dict[str, Any] = {}
    for name, value in attributes.items():
        spec = schema.get(name)
        if spec is None:
            errors.append(f"Unknown field {name!r}")
            continue
        if value is None or value == "":
            continue
        kind = spec["kind"]
        if kind == "categorical":
            if str(value) not in spec["options"]:
                errors.append(f"{name}: {value!r} is not one of the model's known categories")
                continue
            clean[name] = str(value)
        elif kind == "boolean":
            if not isinstance(value, bool):
                errors.append(f"{name}: expected true/false")
                continue
            clean[name] = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{name}: expected a number")
                continue
            if kind == "integer" and float(value) != int(value):
                errors.append(f"{name}: expected a whole number")
                continue
            if spec.get("minimum") is not None and value < spec["minimum"]:
                errors.append(f"{name}: must be >= {spec['minimum']}")
                continue
            clean[name] = int(value) if kind == "integer" else float(value)
    if errors:
        raise ValueError("; ".join(errors))
    return clean


def assess_new_claim(
    state: AppState,
    domain: str,
    attributes: dict[str, Any],
    document: Optional[tuple[bytes, str, int]] = None,
) -> dict[str, Any]:
    model = state.model(domain)
    overrides = validate_attributes(state, domain, attributes)
    claim_key = generate_new_claim_id()
    report = None
    notes = list(claim_schema(state, domain)["notes"])
    if document is not None:
        data, filename, page_index = document
        _, report = analyze_document(data, filename, page_index)
        stream = io.BytesIO(data)
        stream.name = filename
        _, _, _, assessment = assess_new_claim_document(
            domain, model, stream, page_index=page_index, claim_id=claim_key, overrides=overrides
        )
        notes.append(FUSION_NOTE)
    else:
        empty = ForensicUploadResult(ela=None, copy_move=None, metadata=None)
        X_row = build_new_claim_features(domain, claim_key, empty, overrides)
        assessment = assess_claim(domain, model, X_row, DEFAULT_RISK_THRESHOLDS)
    return {
        "domain": domain,
        "claim_key": claim_key,
        "assessment": assessment_to_dict(assessment),
        "forensics": report,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Model insights (saved artifacts only)
# ---------------------------------------------------------------------------


def _read_sidecar(directory, filename: str) -> Optional[dict]:
    path = directory / filename
    if not path.exists():
        return None
    return to_jsonable(json.loads(path.read_text(encoding="utf-8")))


def insights(state: AppState, domain: str) -> dict[str, Any]:
    slot = state.models[domain]
    directory = slot.directory
    model = state.model(domain)
    return {
        "domain": domain,
        "model_dir": str(directory),
        "feature_names": model_feature_names(model),
        "uses_graph_features": uses_graph_features(model),
        "risk_bands": risk_bands(),
        "eval_metrics": to_jsonable(model_loader.load_eval_metrics(directory)),
        "validation_metrics": _read_sidecar(directory, "validation_metrics.json"),
        "test_metrics": to_jsonable(model_loader.load_test_metrics(directory)),
        "threshold_selection": to_jsonable(model_loader.load_threshold_selection(directory)),
        "confusion_matrix": to_jsonable(model_loader.load_confusion_matrix(directory)),
        "roc_curve": to_jsonable(model_loader.load_roc_curve(directory)),
        "pr_curve": to_jsonable(model_loader.load_pr_curve(directory)),
        "probability_diagnostics": _read_sidecar(directory, "probability_diagnostics.json"),
        "data_audit": _read_sidecar(directory, "data_audit.json"),
        "validation_experiment": _read_sidecar(directory, "validation_experiment.json"),
        "provider_grouped_test_metrics": _read_sidecar(directory, "provider_grouped_test_metrics.json"),
        "limitations": LIMITATIONS[domain],
    }
