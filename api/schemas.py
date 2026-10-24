"""Pydantic response models for the InsureSight AI API.

Field names mirror the pipeline's own (`fraud_probability`, `risk_category`,
`top_factors`, ...) so nothing is renamed between `src/` and the browser; the
web UI decides how to label them for investigators.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

RiskCategory = Literal["low", "medium", "high"]
Domain = Literal["health", "motor"]


class RiskBands(BaseModel):
    medium: float = Field(description="Probability at or above which a claim is 'medium' risk")
    high: float = Field(description="Probability at or above which a claim is 'high' risk")


class DomainHealth(BaseModel):
    domain: Domain
    model_loaded: bool
    model_dir: str
    error: Optional[str] = None
    n_features: Optional[int] = None
    uses_graph_features: Optional[bool] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    domains: list[DomainHealth]
    risk_bands: RiskBands


class Factor(BaseModel):
    feature: str
    feature_value: Any = None
    shap_value: float
    source: Literal["model_shap", "document_heuristic"] = Field(
        description=(
            "'model_shap' = TreeExplainer contribution from the trained model; "
            "'document_heuristic' = rule-based document-evidence weight added by "
            "app.claim_view.fuse_assessment_with_document_evidence (not SHAP)"
        )
    )


class Assessment(BaseModel):
    claim_key: Optional[str]
    domain: Domain
    fraud_probability: float
    risk_category: RiskCategory
    risk_category_description: str
    base_value: float
    top_factors: list[Factor]


class ClaimSummary(BaseModel):
    claim_key: str
    fraud_probability: float
    risk_category: RiskCategory
    attributes: dict[str, Any]


class SegmentBreakdown(BaseModel):
    segment: str
    low: int
    medium: int
    high: int


class HistogramBin(BaseModel):
    start: float
    end: float
    count: int


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class OverviewResponse(BaseModel):
    domain: Domain
    sample: dict[str, int]
    total_claims: int
    risk_counts: dict[str, int]
    review_count: int
    review_rate: Optional[float]
    risk_bands: RiskBands
    probability_histogram: list[HistogramBin]
    segment_field: Optional[str]
    segments: list[SegmentBreakdown]
    global_importance: list[FeatureImportance]
    top_claims: list[ClaimSummary]


class ClaimsPage(BaseModel):
    domain: Domain
    sample: dict[str, int]
    total: int
    page: int
    page_size: int
    pages: int
    columns: list[str]
    segment_field: Optional[str]
    segment_options: list[str]
    items: list[ClaimSummary]


class AttributeGroup(BaseModel):
    name: str
    fields: list[str]


class RingEntity(BaseModel):
    type: str
    value: str


class RingEvidence(BaseModel):
    entities: list[RingEntity]
    claims: int = Field(description="Claims in the sample using both entities")
    expected: float = Field(description="Claims expected to share this pair by chance")
    p_value: float
    includes_this_claim: bool


class NetworkBenchmark(BaseModel):
    noise: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    ring_recovery_rate: Optional[float] = None
    naive_precision: Optional[float] = None


class NetworkRisk(BaseModel):
    level: Literal["candidate_ring", "linked", "none"]
    entity_types: list[str]
    method: str
    note: str
    sample_candidate_rings: int
    sample_flagged_claims: int
    sample_claims: int
    ring_id: Optional[str] = None
    ring_size: int = 0
    ring_claims: list[str] = Field(description="Other claims in the candidate ring (capped)")
    evidence: list[RingEvidence]
    benchmark: Optional[NetworkBenchmark] = Field(
        default=None, description="Ring benchmark results for the same detector settings, if generated"
    )


class ClaimDetail(BaseModel):
    domain: Domain
    claim_key: str
    assessment: Assessment
    attributes: dict[str, Any]
    attribute_groups: list[AttributeGroup]
    graph_signals: dict[str, Any]
    model_uses_graph_features: bool
    risk_bands: RiskBands
    network_risk: NetworkRisk
    document_benchmark: Optional[dict[str, Any]] = Field(
        default=None, description="Tampering benchmark verdict detection/false-positive rates, if generated"
    )


ReviewOutcome = Literal["confirmed_suspicious", "cleared", "needs_more_info"]


class ReviewInput(BaseModel):
    outcome: ReviewOutcome
    note: str = Field(default="", max_length=4000)
    reviewer: Optional[str] = Field(default=None, max_length=120)


class ReviewRecord(BaseModel):
    domain: Domain
    claim_key: str
    outcome: ReviewOutcome
    retraining_label: Optional[int] = Field(description="1 confirmed, 0 cleared, null while more info is needed")
    note: str
    reviewer: Optional[str]
    recorded_at: str
    fraud_probability_at_review: Optional[float] = None
    risk_category_at_review: Optional[str] = None
    network_level_at_review: Optional[str] = None
    model_version: Optional[str] = None


class ClaimReview(BaseModel):
    current: Optional[ReviewRecord]
    history: list[ReviewRecord] = Field(description="Every decision for this claim, oldest first")


class RetrainingLabel(BaseModel):
    claim_key: str
    label: int
    recorded_at: str
    model_version: Optional[str] = None


class ReviewSummary(BaseModel):
    domain: Domain
    reviewed_claims: int
    counts: dict[str, int]
    labels: list[RetrainingLabel]
    recent: list[ReviewRecord]


class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    is_focus: bool
    linked_claims: int = Field(description="Number of claims in the loaded sample linked to this node")
    shared: bool = Field(description="Entity node linked to more than one claim in the loaded sample")
    risk_category: Optional[RiskCategory] = None
    fraud_probability: Optional[float] = None
    community_id: Optional[int] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class GraphResponse(BaseModel):
    domain: Domain
    claim_key: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool
    synthetic_entities: list[str] = Field(
        description="Node types whose identifiers are synthetic (not observed in the source data)"
    )


class CopyMoveRegion(BaseModel):
    source: tuple[float, float, float, float]
    target: tuple[float, float, float, float]
    matches: int


class MetadataFlag(BaseModel):
    field: str
    reason: str


class TamperVerdict(BaseModel):
    status: str
    tampering_score: float
    is_tampered: bool
    indicators: list[str]
    summary: str
    details: dict[str, Any]


class ForensicReport(BaseModel):
    filename: str
    is_pdf: bool
    page_index: int
    page_count: int
    image_width: Optional[int]
    image_height: Optional[int]
    original_image: Optional[str] = Field(description="data: URL of the analyzed page/image")
    ela_image: Optional[str]
    copy_move_image: Optional[str]
    ela: Optional[dict[str, Any]]
    copy_move: Optional[dict[str, Any]]
    copy_move_pairs: list[tuple[float, float, float, float]]
    copy_move_regions: list[CopyMoveRegion]
    metadata: Optional[dict[str, Any]]
    metadata_flags: list[MetadataFlag]
    extracted_text: Optional[str]
    verdict: TamperVerdict


class DocumentRescoreResponse(BaseModel):
    domain: Domain
    claim_key: str
    forensics: ForensicReport
    before: Assessment
    after: Assessment
    fusion_note: str


class FieldSpec(BaseModel):
    name: str
    kind: Literal["categorical", "integer", "number", "boolean"]
    options: Optional[list[str]] = None
    default: Any = None
    minimum: Optional[float] = None
    required: bool = False


class ClaimSchema(BaseModel):
    domain: Domain
    fields: list[FieldSpec]
    notes: list[str]


class AssessResponse(BaseModel):
    domain: Domain
    claim_key: str
    assessment: Assessment
    forensics: Optional[ForensicReport]
    notes: list[str]


class InsightsResponse(BaseModel):
    domain: Domain
    model_dir: str
    feature_names: list[str]
    uses_graph_features: bool
    risk_bands: RiskBands
    eval_metrics: Optional[dict[str, Any]]
    validation_metrics: Optional[dict[str, Any]]
    test_metrics: Optional[dict[str, Any]]
    threshold_selection: Optional[dict[str, Any]]
    confusion_matrix: Optional[dict[str, Any]]
    roc_curve: Optional[dict[str, Any]]
    pr_curve: Optional[dict[str, Any]]
    probability_diagnostics: Optional[dict[str, Any]]
    data_audit: Optional[dict[str, Any]]
    validation_experiment: Optional[dict[str, Any]]
    provider_grouped_test_metrics: Optional[dict[str, Any]]
    limitations: list[str]
    tampering_benchmark: Optional[dict[str, Any]] = None
    ring_benchmark: Optional[dict[str, Any]] = None
    workload_curve: Optional[dict[str, Any]] = Field(
        default=None, description="Validation cumulative gain (share reviewed vs. positives caught) from the ROC curve"
    )
