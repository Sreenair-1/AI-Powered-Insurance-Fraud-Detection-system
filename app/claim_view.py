"""Pure orchestration layer between the loaded data/model and a single claim's UI panel.

No `streamlit` import here on purpose: everything in this module is directly
unit-testable, and `app/main.py` is a thin renderer over it. Two flows are
supported for a selected claim:

- `assess_claim`: score the claim using the already-fused feature row from
  `app.data_loader` (forensic features are the "unavailable" placeholder,
  same as what the model was trained on).
- `run_forensic_checks` + `rescore_with_forensics`: if the user uploads a
  document image for the claim, run ELA/copy-move/metadata on it and
  re-fuse *just that claim* with the real forensic result, then re-score.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, cast

import networkx as nx
import numpy as np
import pandas as pd

from app.data_loader import HealthDataset, MotorDataset
from src.data.schemas import MOTOR_ALL_COLUMNS
from src.features.fusion import ForensicResultBundle, build_fused_features
from src.features.structured import CLAIM_KEY_COL, build_motor_structured_features
from src.forensics import (
    CopyMoveImageLoadError,
    CopyMoveResult,
    DocumentTamperingVerdict,
    ELAResult,
    ImageLoadError as ELAImageLoadError,
    MetadataImageLoadError,
    MetadataResult,
    PDFLoadError,
    analyze_metadata,
    analyze_pdf_metadata,
    compute_copy_move,
    compute_ela,
    evaluate_document_tampering,
    render_pdf_page,
)
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.pipeline.inference import (
    DEFAULT_RISK_THRESHOLDS,
    ClaimRiskAssessment,
    RiskThresholds,
    assess_single_health_claim,
    assess_single_motor_claim,
    categorize_risk,
)

HEALTH_DOMAIN = "health"
MOTOR_DOMAIN = "motor"

FORENSIC_ERRORS = (
    ELAImageLoadError,
    CopyMoveImageLoadError,
    MetadataImageLoadError,
    PDFLoadError,
)


class ClaimViewError(ValueError):
    """Raised for an unknown domain or a claim_key not present in the loaded dataset."""


def assess_claim(
    domain: str,
    model: Union[HealthFraudModel, MotorFraudModel],
    X_row: pd.DataFrame,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    top_n: int = 10,
) -> ClaimRiskAssessment:
    """Risk-assess one already-fused claim row against a fitted model.

    Raises:
        ClaimViewError: for an unrecognized `domain`.
        HealthExplainabilityError / MotorModelError: propagated unchanged
            from the underlying pipeline (unfit model, missing feature
            column, etc.) -- see `src/pipeline/inference.py`.
    """
    if domain == HEALTH_DOMAIN:
        return assess_single_health_claim(cast(HealthFraudModel, model), X_row, thresholds, top_n)
    if domain == MOTOR_DOMAIN:
        return assess_single_motor_claim(cast(MotorFraudModel, model), X_row, thresholds, top_n)
    raise ClaimViewError(f"unknown domain {domain!r}; expected {HEALTH_DOMAIN!r} or {MOTOR_DOMAIN!r}")


def claim_row(dataset: Union[HealthDataset, MotorDataset], claim_key: str) -> pd.DataFrame:
    """The single fused feature row for `claim_key`, ready for `assess_claim`."""
    row = dataset.X[dataset.X[CLAIM_KEY_COL] == claim_key]
    if row.empty:
        raise ClaimViewError(f"claim_key {claim_key!r} not found in the loaded dataset")
    return row


MOTOR_INVESTIGATION_COLUMNS = (
    "Make",
    "AccidentArea",
    "Age",
    "PolicyType",
    "Fault",
    "VehicleCategory",
    "Year",
)
HEALTH_INVESTIGATION_COLUMNS = (
    "ClaimID",
    "claim_type",
    "Provider",
    "BeneID",
    "InscClaimAmtReimbursed",
    "AttendingPhysician",
)
GRAPH_SIGNAL_COLUMNS = (
    "has_shared_entity",
    "max_entity_claim_degree",
    "community_id",
    "degree_centrality",
)


def score_loaded_sample(
    model: Union[HealthFraudModel, MotorFraudModel],
    X: pd.DataFrame,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> pd.DataFrame:
    """Score every loaded claim with `predict_proba` only — no SHAP.

    Overview KPIs and the investigation table need risk categories for the
    current sample. Running `assess_claim` (and therefore TreeExplainer) on
    every row would be far too expensive for a Streamlit rerun.
    """
    columns = [CLAIM_KEY_COL, "fraud_probability", "risk_category"]
    if X.empty:
        return pd.DataFrame(columns=columns)
    if CLAIM_KEY_COL not in X.columns:
        raise ClaimViewError(f"fused matrix is missing {CLAIM_KEY_COL}")

    proba = np.asarray(model.predict_proba(X), dtype=float).reshape(-1)
    if len(proba) != len(X):
        raise ClaimViewError("model.predict_proba returned a different row count than X")

    return pd.DataFrame(
        {
            CLAIM_KEY_COL: X[CLAIM_KEY_COL].astype(str).to_numpy(),
            "fraud_probability": proba,
            "risk_category": [categorize_risk(float(p), thresholds) for p in proba],
        }
    )


def build_investigation_table(
    domain: str,
    dataset: Union[HealthDataset, MotorDataset],
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Join sample scores with investigator-facing claim attributes.

    Does not invent fields: only columns present on the loaded claims (or
    fused graph features) are included.
    """
    if domain not in {MOTOR_DOMAIN, HEALTH_DOMAIN}:
        raise ClaimViewError(f"unknown domain {domain!r}; expected {HEALTH_DOMAIN!r} or {MOTOR_DOMAIN!r}")
    if scores.empty:
        return scores.copy()

    display = dataset.claims.copy()
    if domain == MOTOR_DOMAIN:
        display = display.copy()
        display[CLAIM_KEY_COL] = display.index.map(lambda i: f"claim:{i}")
        keep = [c for c in MOTOR_INVESTIGATION_COLUMNS if c in display.columns]
    else:
        if "ClaimID" not in display.columns:
            raise ClaimViewError("health claims table is missing ClaimID")
        display[CLAIM_KEY_COL] = "claim:" + display["ClaimID"].astype(str)
        keep = [c for c in HEALTH_INVESTIGATION_COLUMNS if c in display.columns]

    attr = display[[CLAIM_KEY_COL, *keep]].drop_duplicates(subset=[CLAIM_KEY_COL])
    graph_keep = [c for c in GRAPH_SIGNAL_COLUMNS if c in dataset.X.columns]
    graph = dataset.X[[CLAIM_KEY_COL, *graph_keep]].copy()
    graph[CLAIM_KEY_COL] = graph[CLAIM_KEY_COL].astype(str)

    out = scores.copy()
    out[CLAIM_KEY_COL] = out[CLAIM_KEY_COL].astype(str)
    out = out.merge(attr, on=CLAIM_KEY_COL, how="left")
    out = out.merge(graph, on=CLAIM_KEY_COL, how="left")
    out.insert(1, "domain", domain)
    return out


@dataclass(frozen=True)
class ForensicUploadResult:
    """The outcome of running all three forensic checks on one uploaded document image.

    Any field is `None` if that specific check could not run on the given
    image (corrupt/unsupported file) -- this mirrors how
    `src.features.fusion.ForensicResultBundle` already treats a missing
    check as "unavailable", not an error for the other two checks.
    """

    ela: Optional[ELAResult]
    copy_move: Optional[CopyMoveResult]
    metadata: Optional[MetadataResult]

    @property
    def any_available(self) -> bool:
        return self.ela is not None or self.copy_move is not None or self.metadata is not None


def _is_pdf(document: Any) -> bool:
    """Determine if a document input represents a PDF file or stream."""
    if isinstance(document, (str, Path)):
        return str(document).lower().endswith(".pdf")
    if isinstance(document, (bytes, bytearray)):
        return bytes(document[:5]).startswith(b"%PDF")
    if hasattr(document, "name") and str(getattr(document, "name")).lower().endswith(".pdf"):
        return True
    if hasattr(document, "read") and hasattr(document, "seek"):
        try:
            head = document.read(5)
            document.seek(0)
            return head.startswith(b"%PDF")
        except Exception:
            return False
    return False


def run_forensic_checks(document: Any, page_index: int = 0) -> ForensicUploadResult:
    """Run ELA, copy-move, and metadata analysis on an uploaded claim document (image or PDF).

    `document` is a file path, `PIL.Image`, raw `bytes`, or file-like stream
    (e.g., Streamlit `UploadedFile`). For PDF documents, `page_index` (0-indexed)
    specifies which page to render and analyze for visual forensics (ELA and
    copy-move).

    A load/decode failure in one check does not block the other checks -- each
    is independently optional.
    """
    if _is_pdf(document):
        try:
            rendered_image = render_pdf_page(document, page_index=page_index)
        except (PDFLoadError, ValueError):
            rendered_image = None

        if rendered_image is not None:
            try:
                ela = compute_ela(rendered_image)
            except ELAImageLoadError:
                ela = None
            try:
                copy_move = compute_copy_move(rendered_image)
            except CopyMoveImageLoadError:
                copy_move = None
        else:
            ela = None
            copy_move = None

        try:
            metadata = analyze_pdf_metadata(document, page_index=page_index)
        except (PDFLoadError, ValueError):
            metadata = None

        return ForensicUploadResult(ela=ela, copy_move=copy_move, metadata=metadata)

    try:
        ela = compute_ela(document)
    except ELAImageLoadError:
        ela = None
    except (OSError, ValueError, RuntimeError):
        ela = None

    try:
        copy_move = compute_copy_move(document)
    except CopyMoveImageLoadError:
        copy_move = None
    except (OSError, ValueError, RuntimeError):
        copy_move = None

    try:
        metadata = analyze_metadata(document)
    except MetadataImageLoadError:
        metadata = None
    except (OSError, ValueError, RuntimeError):
        metadata = None
    return ForensicUploadResult(ela=ela, copy_move=copy_move, metadata=metadata)


def fuse_assessment_with_document_evidence(
    base_assessment: ClaimRiskAssessment,
    forensic: ForensicUploadResult,
    verdict: DocumentTamperingVerdict,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    top_n: int = 10,
) -> ClaimRiskAssessment:
    """Fuse tabular/graph model risk with document tampering forensic evidence.

    When an uploaded document is analyzed, combines the tabular baseline fraud
    probability with the document's tampering anomaly score and adds forensic
    contributing factors to the SHAP explanation table.
    """
    if not forensic.any_available or verdict.tampering_score <= 0.0:
        return base_assessment

    p_base = base_assessment.fraud_probability
    t_score = verdict.tampering_score

    # Multi-modal fusion: scale probability upwards when tampering evidence is present
    p_fused = min(0.99, p_base + (1.0 - p_base) * (t_score * 0.85))
    new_risk_category = categorize_risk(p_fused, thresholds)

    # Inject forensic factors into SHAP explanation list
    forensic_factors: list[dict[str, object]] = []

    if forensic.metadata is not None:
        if forensic.metadata.features.software_editing_indicator:
            soft_name = forensic.metadata.structured_metadata.get("software", "Editing Software")
            forensic_factors.append({
                "feature": "Document Forensics: Editing Software Signature",
                "feature_value": soft_name,
                "shap_value": 1.50 * t_score,
            })
        if forensic.metadata.features.datetime_original_modified_mismatch:
            forensic_factors.append({
                "feature": "Document Forensics: Timestamp Modification Mismatch",
                "feature_value": "Modified post-creation",
                "shap_value": 0.85 * t_score,
            })
        if forensic.metadata.features.exif_malformed:
            forensic_factors.append({
                "feature": "Document Forensics: Corrupted / Malformed Metadata",
                "feature_value": "Malformed IFD block",
                "shap_value": 0.50 * t_score,
            })

    if forensic.copy_move is not None and forensic.copy_move.potential_duplicate_regions_found:
        cm_matches = forensic.copy_move.features.num_filtered_matches
        forensic_factors.append({
            "feature": "Document Forensics: Copy-Move Cloned Regions",
            "feature_value": f"{cm_matches} matched keypoints",
            "shap_value": 1.30 * t_score,
        })

    if forensic.ela is not None and (
        forensic.ela.features.high_error_pixel_ratio > 0.03 or forensic.ela.features.p95_error > 45.0
    ):
        err_ratio = forensic.ela.features.high_error_pixel_ratio
        forensic_factors.append({
            "feature": "Document Forensics: ELA Compression Splicing Discrepancy",
            "feature_value": f"{err_ratio:.1%} high-error pixels",
            "shap_value": 1.10 * t_score,
        })

    # Combine with base SHAP factors and sort by absolute contribution
    all_factors = list(base_assessment.top_factors) + forensic_factors
    all_factors.sort(key=lambda f: abs(float(f.get("shap_value", 0.0))), reverse=True)
    updated_top_factors = tuple(all_factors[:top_n])

    return ClaimRiskAssessment(
        claim_key=base_assessment.claim_key,
        domain=base_assessment.domain,
        fraud_probability=p_fused,
        risk_category=new_risk_category,
        base_value=base_assessment.base_value,
        top_factors=updated_top_factors,
    )


def rescore_with_forensics(
    domain: str,
    model: Union[HealthFraudModel, MotorFraudModel],
    dataset: Union[HealthDataset, MotorDataset],
    claim_key: str,
    forensic: ForensicUploadResult,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    top_n: int = 10,
) -> ClaimRiskAssessment:
    """Re-fuse `claim_key` with a real forensic bundle and re-score it.

    Rebuilds only this one claim's fused feature row (structured + the same
    cached graph + the uploaded document's forensic result) rather than the
    whole sampled dataset, then runs it through the normal risk-assessment
    path.

    Raises:
        ClaimViewError: if `claim_key` is not in `dataset.structured_df`, or
            `domain` is unrecognized.
    """
    structured_row = dataset.structured_df[dataset.structured_df[CLAIM_KEY_COL] == claim_key]
    if structured_row.empty:
        raise ClaimViewError(f"claim_key {claim_key!r} not found in the loaded dataset")

    bundle = ForensicResultBundle(ela=forensic.ela, copy_move=forensic.copy_move, metadata=forensic.metadata)
    X_row = build_fused_features(structured_row, dataset.graph, forensic_results={claim_key: bundle})
    base_assessment = assess_claim(domain, model, X_row, thresholds, top_n)
    verdict = evaluate_document_tampering(ela=forensic.ela, copy_move=forensic.copy_move, metadata=forensic.metadata)
    return fuse_assessment_with_document_evidence(base_assessment, forensic, verdict, thresholds, top_n)


def generate_new_claim_id(prefix: str = "NEW") -> str:
    """Generate a clean, unique claim identifier for a newly submitted document."""
    suffix = secrets.token_hex(3).upper()
    return f"claim:{prefix}-{suffix}"


def build_new_claim_features(
    domain: str,
    claim_key: str,
    forensic: ForensicUploadResult,
    overrides: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Construct a full fused feature matrix for a brand new claim.

    Uses domain-standard baseline features (which can be overridden with
    `overrides`), builds an isolated 1-node graph (neutral centrality = 0.0,
    no shared entities), and fuses the real uploaded document's forensic result.

    Args:
        domain: 'motor' or 'health'
        claim_key: Unique claim identifier (e.g. 'claim:NEW-849201')
        forensic: The ForensicUploadResult from running checks on the document
        overrides: Optional dict of specific feature fields to override

    Returns:
        One-row pd.DataFrame ready for risk model inference.
    """
    user_vals = overrides or {}
    bundle = ForensicResultBundle(
        ela=forensic.ela,
        copy_move=forensic.copy_move,
        metadata=forensic.metadata,
    )

    G = nx.Graph()
    G.add_node(claim_key, node_type="claim")

    if domain == MOTOR_DOMAIN:
        raw_row = {c: 0 for c in MOTOR_ALL_COLUMNS}
        raw_row["Make"] = user_vals.get("Make", "Honda")
        raw_row["AccidentArea"] = user_vals.get("AccidentArea", "Urban")
        raw_row["Sex"] = user_vals.get("Sex", "Male")
        raw_row["Age"] = user_vals.get("Age", 35)
        raw_row["PolicyType"] = user_vals.get("PolicyType", "Sedan - All Perils")
        raw_row["BasePolicy"] = user_vals.get("BasePolicy", "All Perils")
        raw_row["VehicleCategory"] = user_vals.get("VehicleCategory", "Sedan")
        raw_row["VehiclePrice"] = user_vals.get("VehiclePrice", "20000 to 29000")
        raw_row["Deductible"] = user_vals.get("Deductible", 400)
        raw_row["DriverRating"] = user_vals.get("DriverRating", 1)
        raw_row["PastNumberOfClaims"] = user_vals.get("PastNumberOfClaims", "none")
        raw_row["AgeOfVehicle"] = user_vals.get("AgeOfVehicle", "3 years")
        raw_row["PoliceReportFiled"] = user_vals.get("PoliceReportFiled", "No")
        raw_row["WitnessPresent"] = user_vals.get("WitnessPresent", "No")
        raw_row["AddressChange_Claim"] = user_vals.get("AddressChange_Claim", "no change")
        raw_row["NumberOfCars"] = user_vals.get("NumberOfCars", "1 vehicle")
        raw_row["Year"] = user_vals.get("Year", 1995)
        raw_row["GarageID"] = user_vals.get("GarageID", "GRG00001")
        raw_row["BankAccountID"] = user_vals.get("BankAccountID", "BNK00001")

        struct_df = build_motor_structured_features(pd.DataFrame([raw_row]))
        struct_df[CLAIM_KEY_COL] = claim_key
        return build_fused_features(struct_df, G, forensic_results={claim_key: bundle})

    if domain == HEALTH_DOMAIN:
        health_row = {
            CLAIM_KEY_COL: claim_key,
            "claim_type": user_vals.get("claim_type", "inpatient"),
            "num_diagnosis_codes": user_vals.get("num_diagnosis_codes", 2),
            "num_procedure_codes": user_vals.get("num_procedure_codes", 1),
            "is_deceased": user_vals.get("is_deceased", False),
            "chronic_condition_count": user_vals.get("chronic_condition_count", 1),
            "has_renal_disease": user_vals.get("has_renal_disease", False),
        }
        df_health = pd.DataFrame([health_row])
        df_health["claim_type"] = df_health["claim_type"].astype("category")
        return build_fused_features(df_health, G, forensic_results={claim_key: bundle})

    raise ClaimViewError(f"unknown domain {domain!r}; expected {HEALTH_DOMAIN!r} or {MOTOR_DOMAIN!r}")


def assess_new_claim_document(
    domain: str,
    model: Union[HealthFraudModel, MotorFraudModel],
    document: Any,
    page_index: int = 0,
    claim_id: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    top_n: int = 10,
) -> tuple[str, ForensicUploadResult, DocumentTamperingVerdict, ClaimRiskAssessment]:
    """Execute complete standalone forensic and tampering assessment for a new claim document.

    Generates a new unique claim ID, executes Error Level Analysis, Copy-Move
    Detection, and Metadata Analysis, synthesizes the Document Tampering Verdict,
    fuses the features, and computes the model's fraud risk assessment.

    Args:
        domain: 'motor' or 'health'
        model: Fitted HealthFraudModel or MotorFraudModel
        document: PDF or image input
        page_index: Page to analyze if multi-page PDF
        claim_id: Optional claim ID (auto-generated if None)
        overrides: Optional user-specified claim attributes
        thresholds: RiskThresholds
        top_n: Top SHAP factors to return

    Returns:
        tuple of (claim_id, forensic_result, tampering_verdict, risk_assessment)
    """
    new_claim_key = claim_id or generate_new_claim_id()
    forensic = run_forensic_checks(document, page_index=page_index)
    verdict = evaluate_document_tampering(
        ela=forensic.ela,
        copy_move=forensic.copy_move,
        metadata=forensic.metadata,
    )
    X_row = build_new_claim_features(domain, new_claim_key, forensic, overrides)
    base_assessment = assess_claim(domain, model, X_row, thresholds, top_n)
    fused_assessment = fuse_assessment_with_document_evidence(
        base_assessment, forensic, verdict, thresholds, top_n
    )
    return new_claim_key, forensic, verdict, fused_assessment
