"""End-to-end per-claim fraud risk inference pipeline.

Wires together the four pieces that already exist independently:

- a claim-keyed fused feature matrix (`src.features.fusion`,
  `fuse_health_claims()` / `fuse_motor_claims()`);
- an already-fitted `HealthFraudModel` / `MotorFraudModel`
  (`src.models.health` / `src.models.motor`);
- shared evaluation metrics (`src.models.evaluation`);
- SHAP explainability (`src.explainability.health` /
  `src.explainability.motor`).

into one call that produces the shape a decision-support UI needs: a
predicted fraud probability, a risk category, and the SHAP factors that
drove that prediction, per claim.

This module does not fit models and does not build feature matrices itself
- like `src.features.fusion`, it is purely an orchestration/assembly layer
over already-fitted models and already-fused feature matrices. It does not
reconcile `src.explainability.health` / `src.explainability.motor` onto the
shared `src.explainability.common` layer (see `.claude/NEXT_TASKS.md`); it
simply reuses whichever per-model explainability module already exists for
each domain.

Per CLAUDE.md section 9, `risk_category` is a decision-support label
("low"/"medium"/"high"), never a fraud determination — the same caveat
applies to every value this module returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.explainability.health import HealthShapExplanation, explain_health_claims
from src.explainability.motor import MotorShapExplanation, explain_motor_claims
from src.models.evaluation import ClassificationMetrics, evaluate_model
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel

HEALTH_DOMAIN = "health"
MOTOR_DOMAIN = "motor"

# Human-readable, non-accusatory labels for each risk category, matching the
# language CLAUDE.md section 9 requires ("elevated risk", "flagged for
# investigation") rather than any claim of proven fraud.
RISK_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "low": "Low risk",
    "medium": "Elevated risk - review recommended",
    "high": "High risk - flagged for investigation",
}


class PipelineError(ValueError):
    """Raised for invalid input/state anywhere in the inference pipeline."""


@dataclass(frozen=True)
class RiskThresholds:
    """Probability cutoffs that turn a fraud probability into a risk category.

    `probability >= high` -> `"high"`; `probability >= medium` -> `"medium"`;
    otherwise `"low"`. Thresholds are a decision-support convenience, not a
    fitted or validated operating point.
    """

    medium: float = 0.3
    high: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 <= self.medium <= 1.0:
            raise PipelineError(f"medium threshold must be in [0, 1], got {self.medium}")
        if not 0.0 <= self.high <= 1.0:
            raise PipelineError(f"high threshold must be in [0, 1], got {self.high}")
        if not self.medium < self.high:
            raise PipelineError(
                f"medium threshold ({self.medium}) must be less than high threshold ({self.high})"
            )


DEFAULT_RISK_THRESHOLDS = RiskThresholds()


def categorize_risk(probability: float, thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS) -> str:
    """Map a fraud probability to a `"low"` / `"medium"` / `"high"` risk category.

    Raises:
        PipelineError: if `probability` is outside `[0, 1]`.
    """
    if not 0.0 <= probability <= 1.0:
        raise PipelineError(f"probability must be in [0, 1], got {probability}")
    if probability >= thresholds.high:
        return "high"
    if probability >= thresholds.medium:
        return "medium"
    return "low"


def _top_factors_from_row(
    feature_names: Sequence[str],
    shap_row: np.ndarray,
    raw_row: pd.Series,
    top_n: int,
) -> tuple[dict[str, object], ...]:
    """Build a `top_n`-length, |SHAP|-descending list of per-feature contributions.

    `feature_value` is read from `raw_row` (the caller's own fused input),
    not from whatever internal representation the model/SHAP layer used
    (e.g. motor's fit-time categorical re-encoding) - the value shown should
    be the one an investigator recognizes from the claim, not an encoding
    detail.
    """
    order = np.argsort(-np.abs(shap_row))[:top_n]
    factors = []
    for i in order:
        feature = feature_names[i]
        factors.append(
            {
                "feature": feature,
                "feature_value": raw_row[feature] if feature in raw_row.index else None,
                "shap_value": float(shap_row[i]),
            }
        )
    return tuple(factors)


@dataclass(frozen=True)
class ClaimRiskAssessment:
    """The full decision-support output for one claim.

    `risk_category` and `top_factors` describe model behavior, not a fraud
    determination (CLAUDE.md section 9) - a large positive `shap_value`
    means the model's prediction was sensitive to that feature, not that the
    feature proves fraud.
    """

    claim_key: Optional[str]
    domain: str
    fraud_probability: float
    risk_category: str
    base_value: float
    top_factors: tuple[dict[str, object], ...]

    @property
    def risk_category_description(self) -> str:
        return RISK_CATEGORY_DESCRIPTIONS[self.risk_category]

    def to_ui_summary(self) -> dict[str, object]:
        """Plain dict/list shape for a Streamlit page - no pandas/numpy types."""
        return {
            "claim_key": self.claim_key,
            "domain": self.domain,
            "fraud_probability": self.fraud_probability,
            "risk_category": self.risk_category,
            "risk_category_description": self.risk_category_description,
            "base_value": self.base_value,
            "top_factors": [dict(f) for f in self.top_factors],
        }


def assess_health_claims(
    model: HealthFraudModel,
    X: pd.DataFrame,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    top_n: int = 10,
) -> list[ClaimRiskAssessment]:
    """Risk-assess every row of `X` against a fitted `HealthFraudModel`.

    Raises:
        HealthExplainabilityError: if `model` is unfit or `X` is missing a
            feature column seen at fit time (propagated from
            `explain_health_claims`).
    """
    explanation: HealthShapExplanation = explain_health_claims(model, X)
    assessments = []
    for i in range(explanation.n_samples):
        proba = float(explanation.predicted_proba[i])
        claim_key = explanation.claim_keys[i] if explanation.claim_keys is not None else None
        top_factors = _top_factors_from_row(
            explanation.feature_names, explanation.shap_values[i], X.iloc[i], top_n
        )
        assessments.append(
            ClaimRiskAssessment(
                claim_key=claim_key,
                domain=HEALTH_DOMAIN,
                fraud_probability=proba,
                risk_category=categorize_risk(proba, thresholds),
                base_value=explanation.base_value,
                top_factors=top_factors,
            )
        )
    return assessments


def assess_motor_claims(
    model: MotorFraudModel,
    X: pd.DataFrame,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    top_n: int = 10,
) -> list[ClaimRiskAssessment]:
    """Risk-assess every row of `X` against a fitted `MotorFraudModel`.

    Raises:
        MotorModelError: if `model` is unfit or `X` fails the same
            validation `predict`/`predict_proba` apply (propagated from
            `explain_motor_claims` / `model.predict_proba`).
    """
    explanation: MotorShapExplanation = explain_motor_claims(model, X)
    proba = model.predict_proba(X)
    assessments = []
    for i, claim_key in enumerate(explanation.claim_keys):
        p = float(proba[i])
        top_factors = _top_factors_from_row(
            explanation.feature_names, explanation.shap_values[i], X.iloc[i], top_n
        )
        assessments.append(
            ClaimRiskAssessment(
                claim_key=claim_key,
                domain=MOTOR_DOMAIN,
                fraud_probability=p,
                risk_category=categorize_risk(p, thresholds),
                base_value=explanation.base_value,
                top_factors=top_factors,
            )
        )
    return assessments


def assess_single_health_claim(
    model: HealthFraudModel,
    X_row: pd.DataFrame,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    top_n: int = 10,
) -> ClaimRiskAssessment:
    """Convenience wrapper: risk-assess exactly one health claim.

    Raises:
        PipelineError: if `X_row` does not contain exactly one row.
    """
    if len(X_row) != 1:
        raise PipelineError(f"X_row must contain exactly one row, got {len(X_row)}")
    return assess_health_claims(model, X_row, thresholds, top_n)[0]


def assess_single_motor_claim(
    model: MotorFraudModel,
    X_row: pd.DataFrame,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    top_n: int = 10,
) -> ClaimRiskAssessment:
    """Convenience wrapper: risk-assess exactly one motor claim.

    Raises:
        PipelineError: if `X_row` does not contain exactly one row.
    """
    if len(X_row) != 1:
        raise PipelineError(f"X_row must contain exactly one row, got {len(X_row)}")
    return assess_motor_claims(model, X_row, thresholds, top_n)[0]


@dataclass(frozen=True)
class PipelineResult:
    """The full output of running a domain's inference pipeline over a batch.

    `metrics` is `None` whenever no ground-truth `y_true` was supplied (e.g.
    scoring claims with an unknown outcome) - metrics describe model
    behavior on a labeled split, not any individual claim.
    """

    domain: str
    assessments: tuple[ClaimRiskAssessment, ...]
    metrics: Optional[ClassificationMetrics]

    def to_ui_summary(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "assessments": [a.to_ui_summary() for a in self.assessments],
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
        }


def run_health_pipeline(
    model: HealthFraudModel,
    X: pd.DataFrame,
    y_true: Optional[Union[pd.Series, np.ndarray]] = None,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    metrics_threshold: float = 0.5,
    top_n: int = 10,
) -> PipelineResult:
    """Run the full health inference pipeline: risk-assess `X`, optionally evaluate against `y_true`.

    `y_true`, if given, must be the same length/order as `X` (e.g. the `y`
    `fuse_health_claims()` returned for this exact `X`) - it is scored with
    `src.models.evaluation.evaluate_model`, never derived from the SHAP
    output.
    """
    assessments = assess_health_claims(model, X, thresholds, top_n)
    metrics = (
        evaluate_model(model, X, y_true, threshold=metrics_threshold) if y_true is not None else None
    )
    return PipelineResult(domain=HEALTH_DOMAIN, assessments=tuple(assessments), metrics=metrics)


def run_motor_pipeline(
    model: MotorFraudModel,
    X: pd.DataFrame,
    y_true: Optional[Union[pd.Series, np.ndarray]] = None,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
    metrics_threshold: float = 0.5,
    top_n: int = 10,
) -> PipelineResult:
    """Run the full motor inference pipeline: risk-assess `X`, optionally evaluate against `y_true`.

    `y_true`, if given, must be the same length/order as `X` (e.g. the `y`
    `fuse_motor_claims()` returned for this exact `X`) - it is scored with
    `src.models.evaluation.evaluate_model`, never derived from the SHAP
    output.
    """
    assessments = assess_motor_claims(model, X, thresholds, top_n)
    metrics = (
        evaluate_model(model, X, y_true, threshold=metrics_threshold) if y_true is not None else None
    )
    return PipelineResult(domain=MOTOR_DOMAIN, assessments=tuple(assessments), metrics=metrics)
