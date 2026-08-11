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

from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import pandas as pd

from app.data_loader import HealthDataset, MotorDataset
from src.features.fusion import ForensicResultBundle, build_fused_features
from src.features.structured import CLAIM_KEY_COL
from src.forensics import (
    CopyMoveImageLoadError,
    CopyMoveResult,
    ELAResult,
    ImageLoadError as ELAImageLoadError,
    MetadataImageLoadError,
    MetadataResult,
    analyze_metadata,
    compute_copy_move,
    compute_ela,
)
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.pipeline.inference import (
    DEFAULT_RISK_THRESHOLDS,
    ClaimRiskAssessment,
    RiskThresholds,
    assess_single_health_claim,
    assess_single_motor_claim,
)

HEALTH_DOMAIN = "health"
MOTOR_DOMAIN = "motor"

FORENSIC_ERRORS = (ELAImageLoadError, CopyMoveImageLoadError, MetadataImageLoadError)


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


def run_forensic_checks(image: Any) -> ForensicUploadResult:
    """Run ELA, copy-move, and metadata analysis on an uploaded claim document image.

    `image` is a path or `PIL.Image` (anything `src.forensics.*` already
    accepts). A load/decode failure in one check does not block the other
    two -- each is independently optional.
    """
    try:
        ela = compute_ela(image)
    except ELAImageLoadError:
        ela = None
    try:
        copy_move = compute_copy_move(image)
    except CopyMoveImageLoadError:
        copy_move = None
    try:
        metadata = analyze_metadata(image)
    except MetadataImageLoadError:
        metadata = None
    return ForensicUploadResult(ela=ela, copy_move=copy_move, metadata=metadata)


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
    return assess_claim(domain, model, X_row, thresholds, top_n)
