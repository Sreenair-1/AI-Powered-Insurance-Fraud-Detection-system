"""Explainability layer — SHAP-based explanations for fraud risk models.

Re-exports the shared model-agnostic SHAP layer (`src.explainability.common`)
plus the `HealthFraudModel`/`MotorFraudModel`-specific wrappers built on top
of it (`src.explainability.health` / `src.explainability.motor`) — the full
public API of this package in one place.
"""
from src.explainability.common import (
    ExplainabilityError,
    GlobalExplanation,
    LocalExplanation,
    create_tree_explainer,
    explain_global,
    explain_local,
    explain_one,
    prepare_features_for_shap,
)
from src.explainability.health import (
    HealthExplainabilityError,
    HealthShapExplanation,
    explain_health_claims,
    explain_single_health_claim,
)
from src.explainability.motor import (
    MotorShapExplanation,
    explain_motor_claims,
)

__all__ = [
    "ExplainabilityError",
    "GlobalExplanation",
    "LocalExplanation",
    "create_tree_explainer",
    "explain_global",
    "explain_local",
    "explain_one",
    "prepare_features_for_shap",
    "HealthExplainabilityError",
    "HealthShapExplanation",
    "explain_health_claims",
    "explain_single_health_claim",
    "MotorShapExplanation",
    "explain_motor_claims",
]
