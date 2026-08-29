"""Document tampering evaluation and integrity assessment.

Synthesizes multiple forensic evidence channels — Error Level Analysis (ELA),
Copy-Move Keypoint Matching, and Document Catalog/EXIF Metadata Analysis — into
an interpretable document tampering report with severity levels, confidence,
and actionable indicator summaries.

Limitations (read before use):
- Tampering scoring is an evidentiary assessment of digital anomalies, not legal
  proof of document forgery or fraud.
- Legitimate document workflows (e.g., converting Word to PDF, scanning with
  automatic cleanup, flattening forms) can trigger individual forensic flags.
- This module provides decision-support evidence to be weighed alongside
  policy and relationship analytics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.forensics.copy_move import CopyMoveResult
from src.forensics.ela import ELAResult
from src.forensics.metadata import MetadataResult


@dataclass(frozen=True)
class DocumentTamperingVerdict:
    """Consolidated document tampering and integrity assessment."""

    status: str  # 'authentic', 'suspicious', 'tampered'
    tampering_score: float  # 0.0 (clean) to 1.0 (high tampering likelihood)
    is_tampered: bool  # True if status in ('suspicious', 'tampered')
    tampering_indicators: list[str]
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_document_tampering(
    ela: Optional[ELAResult] = None,
    copy_move: Optional[CopyMoveResult] = None,
    metadata: Optional[MetadataResult] = None,
) -> DocumentTamperingVerdict:
    """Evaluate multiple forensic signals to produce a document tampering verdict.

    Args:
        ela: Optional ELAResult from compute_ela.
        copy_move: Optional CopyMoveResult from compute_copy_move.
        metadata: Optional MetadataResult from analyze_metadata or analyze_pdf_metadata.

    Returns:
        DocumentTamperingVerdict with tampering status, score, indicators, and summary.
    """
    indicators: list[str] = []
    score = 0.0
    details: dict[str, Any] = {
        "ela_anomaly": False,
        "copy_move_anomaly": False,
        "metadata_anomaly": False,
    }

    # 1. Metadata Analysis
    if metadata is not None:
        feats = metadata.features
        if feats.software_editing_indicator:
            software_tag = metadata.structured_metadata.get("software", "Unknown editor")
            indicators.append(
                f"Image/PDF editing software signature detected in document metadata: '{software_tag}'."
            )
            score += 0.35
            details["metadata_anomaly"] = True

        if feats.datetime_original_modified_mismatch:
            orig = metadata.structured_metadata.get("datetime_original", "N/A")
            mod = metadata.structured_metadata.get("datetime_modified", "N/A")
            indicators.append(
                f"Document modification timestamp mismatch: created ({orig}) vs. modified ({mod})."
            )
            score += 0.20
            details["metadata_anomaly"] = True

        if feats.exif_malformed:
            indicators.append("Document structural metadata contains malformed or corrupted blocks.")
            score += 0.10
            details["metadata_anomaly"] = True

    # 2. Copy-Move Forgery Detection
    if copy_move is not None:
        if copy_move.potential_duplicate_regions_found:
            cm_feats = copy_move.features
            num_matches = cm_feats.num_filtered_matches
            cluster_sz = cm_feats.max_cluster_size
            indicators.append(
                f"Potential duplicated/cloned regions detected ({num_matches} matching keypoint pairs, "
                f"largest spatial offset cluster: {cluster_sz})."
            )
            score += 0.30
            if cluster_sz >= 3:
                score += 0.10  # Coherent block duplication has higher confidence
            details["copy_move_anomaly"] = True

    # 3. Error Level Analysis (ELA)
    if ela is not None:
        ela_feats = ela.features
        # High error pixel ratio > 3% or very high max difference
        if ela_feats.high_error_pixel_ratio > 0.03:
            indicators.append(
                f"Elevated compression error variance detected (high-error pixel ratio: "
                f"{ela_feats.high_error_pixel_ratio:.1%}), suggesting localized digital splicing."
            )
            score += min(0.30, ela_feats.high_error_pixel_ratio * 3.0)
            details["ela_anomaly"] = True
        elif ela_feats.p95_error > 45.0:
            indicators.append(
                f"Localized pixel recompression variance detected (95th-percentile error: {ela_feats.p95_error:.1f})."
            )
            score += 0.15
            details["ela_anomaly"] = True

    # Normalize score
    tampering_score = min(1.0, max(0.0, score))

    if tampering_score >= 0.50:
        status = "tampered"
        summary = (
            "High likelihood of document manipulation: multiple conclusive forensic anomalies "
            "or digital editing tool signatures detected."
        )
    elif tampering_score >= 0.20:
        status = "suspicious"
        summary = (
            "Elevated forensic anomalies detected: document exhibits revisions or localized "
            "irregularities that warrant detailed investigator review."
        )
    else:
        status = "authentic"
        summary = (
            "No significant tampering anomalies detected. Document appears authentic based on "
            "analyzed compression, keypoints, and metadata."
        )

    return DocumentTamperingVerdict(
        status=status,
        tampering_score=tampering_score,
        is_tampered=(status != "authentic"),
        tampering_indicators=indicators,
        summary=summary,
        details=details,
    )
