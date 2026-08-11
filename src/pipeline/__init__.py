"""Inference pipeline: fused features + trained model + evaluation + SHAP,
combined into one per-claim risk assessment. See `src/pipeline/inference.py`.
"""
from src.pipeline.inference import (
    DEFAULT_RISK_THRESHOLDS,
    HEALTH_DOMAIN,
    MOTOR_DOMAIN,
    RISK_CATEGORY_DESCRIPTIONS,
    ClaimRiskAssessment,
    PipelineError,
    PipelineResult,
    RiskThresholds,
    assess_health_claims,
    assess_motor_claims,
    assess_single_health_claim,
    assess_single_motor_claim,
    categorize_risk,
    run_health_pipeline,
    run_motor_pipeline,
)

__all__ = [
    "HEALTH_DOMAIN",
    "MOTOR_DOMAIN",
    "RISK_CATEGORY_DESCRIPTIONS",
    "DEFAULT_RISK_THRESHOLDS",
    "RiskThresholds",
    "PipelineError",
    "ClaimRiskAssessment",
    "PipelineResult",
    "categorize_risk",
    "assess_health_claims",
    "assess_motor_claims",
    "assess_single_health_claim",
    "assess_single_motor_claim",
    "run_health_pipeline",
    "run_motor_pipeline",
]
