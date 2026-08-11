"""Feature adapters + fusion: structured, graph, and forensic features joined
into one claim-keyed, model-ready matrix.

See docs/feature_fusion.md for the `claim_key` join contract and the fused
matrix's leakage-exclusion rules.
"""
from src.features.forensic import (
    FORENSIC_FEATURE_NAMES,
    extract_copy_move_features,
    extract_ela_features,
    extract_forensic_features,
    extract_metadata_features,
)
from src.features.fusion import (
    TARGET_COL,
    ForensicResultBundle,
    build_forensic_claim_features,
    build_fused_features,
    build_health_target,
    build_motor_target,
    fuse_health_claims,
    fuse_motor_claims,
)
from src.features.graph import build_graph_claim_features
from src.features.structured import (
    CLAIM_KEY_COL,
    build_health_structured_features,
    build_motor_structured_features,
)

__all__ = [
    "CLAIM_KEY_COL",
    "TARGET_COL",
    "FORENSIC_FEATURE_NAMES",
    "build_motor_structured_features",
    "build_health_structured_features",
    "build_graph_claim_features",
    "extract_ela_features",
    "extract_copy_move_features",
    "extract_metadata_features",
    "extract_forensic_features",
    "ForensicResultBundle",
    "build_forensic_claim_features",
    "build_fused_features",
    "build_motor_target",
    "build_health_target",
    "fuse_motor_claims",
    "fuse_health_claims",
]
