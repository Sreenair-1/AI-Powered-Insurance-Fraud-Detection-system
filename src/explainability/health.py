"""SHAP explainability for the health fraud risk model.

Thin `HealthFraudModel`-specific wrapper around the shared, model-agnostic
SHAP layer in `src.explainability.common`. Never retrains or refits the
model — every function here only reads `model.model` (the fitted
`XGBClassifier`) and `model.feature_names_`, and delegates explainer
construction, feature alignment, and SHAP computation to
`create_tree_explainer()` / `explain_local()` / `explain_one()`
(`src/explainability/common.py`) rather than calling `shap.TreeExplainer`
directly or re-implementing feature alignment.

SHAP values are computed in the model's native margin (log-odds) output
space, so `base_value + sum(shap_values for a row) == raw model margin`
for that row (the standard additive SHAP guarantee for
`shap.TreeExplainer` on a tree-ensemble classifier — see `docs/shap.md`).
Predicted probabilities are reported separately (via `model.predict_proba`)
for UI display, not derived from the SHAP values themselves.

Feature alignment is exactly `common.prepare_features_for_shap`'s
contract: `claim_key` and any known leakage/target column (including
`claim_fraud_label_provider_derived`) are dropped, then the remaining
columns are reindexed onto `model.feature_names_` — the exact column set
and order fixed at fit time.

This module supports an academic decision-support system, not an
autonomous fraud adjudication system: SHAP values explain what the
model's prediction was sensitive to for a given claim, not proof that
the claim is fraudulent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.explainability.common import (
    ExplainabilityError,
    create_tree_explainer,
    explain_local,
    explain_one,
)
from src.features.structured import CLAIM_KEY_COL
from src.models.health import HealthFraudModel


class HealthExplainabilityError(ExplainabilityError):
    """Raised for invalid input/state when explaining a HealthFraudModel.

    Subclasses `src.explainability.common.ExplainabilityError` (rather than
    `ValueError` directly) so this module's errors also satisfy
    `except ExplainabilityError` at a call site that doesn't care which
    model produced them.
    """


def _require_fitted(model: HealthFraudModel) -> list[str]:
    """Return `model.feature_names_` as a list, or raise if unfitted."""
    if model.model is None or model.feature_names_ is None:
        raise HealthExplainabilityError(
            "HealthFraudModel must be fit() before it can be explained"
        )
    return list(model.feature_names_)


def _explainer_for(model: HealthFraudModel):
    """`create_tree_explainer(model.model)`, with errors re-typed to `HealthExplainabilityError`."""
    try:
        return create_tree_explainer(model.model)
    except ExplainabilityError as exc:
        raise HealthExplainabilityError(str(exc)) from exc


@dataclass(frozen=True)
class HealthShapExplanation:
    """SHAP explanation for one or more health claims.

    `shap_values[i, j]` is the contribution of `feature_names[j]` to row
    `i`'s prediction (margin/log-odds space); `base_value` is the shared
    expected model output. `predicted_proba[i]` is `model.predict_proba`'s
    output for row `i`, reported for UI convenience — not derived from the
    SHAP values.
    """

    feature_names: tuple[str, ...]
    shap_values: np.ndarray
    base_value: float
    predicted_proba: np.ndarray
    claim_keys: Optional[tuple[str, ...]]

    @property
    def n_samples(self) -> int:
        return self.shap_values.shape[0]

    def global_importance(self) -> pd.DataFrame:
        """Mean absolute SHAP value per feature, sorted most-to-least important."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        out = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap_value": mean_abs})
        return out.sort_values("mean_abs_shap_value", ascending=False).reset_index(drop=True)

    def claim_contributions(self, row_index: int) -> pd.DataFrame:
        """Per-feature SHAP contributions for one row, sorted by |contribution|."""
        if not (0 <= row_index < self.n_samples):
            raise HealthExplainabilityError(
                f"row_index {row_index} out of range for {self.n_samples} explained row(s)"
            )
        out = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_value": self.shap_values[row_index],
            }
        )
        return out.reindex(out["shap_value"].abs().sort_values(ascending=False).index).reset_index(
            drop=True
        )

    def to_ui_summary(self, top_n: int = 10) -> dict:
        """Compact, JSON-serializable-friendly summary for later UI consumption.

        Not a UI implementation itself — just a plain-dict/list shape a
        Streamlit page can render directly without importing this module's
        dataclasses/pandas types.
        """
        global_df = self.global_importance().head(top_n)
        return {
            "n_samples": self.n_samples,
            "base_value": self.base_value,
            "top_global_features": [
                {"feature": row.feature, "mean_abs_shap_value": float(row.mean_abs_shap_value)}
                for row in global_df.itertuples(index=False)
            ],
        }


def explain_health_claims(model: HealthFraudModel, X: pd.DataFrame) -> HealthShapExplanation:
    """Compute a SHAP explanation for every row of `X` against a fitted `model`.

    `X` must contain (at least) every column in `model.feature_names_`; an
    optional `claim_key` column, if present, is dropped before SHAP and
    carried through separately on the returned explanation. Deterministic:
    `shap.TreeExplainer` on a fixed fitted tree ensemble with fixed input
    produces identical output on repeated calls.

    Raises:
        HealthExplainabilityError: if `model` has not been fit, or `X` is
            missing a feature column the model was fit on.
    """
    feature_names = _require_fitted(model)
    explainer = _explainer_for(model)

    try:
        local_explanations = explain_local(explainer, X, feature_names)
    except ExplainabilityError as exc:
        raise HealthExplainabilityError(str(exc)) from exc

    shap_values = np.array(
        [[le.feature_contributions[name] for name in feature_names] for le in local_explanations]
    )
    base_value = local_explanations[0].base_value

    predicted_proba = model.predict_proba(X)
    claim_keys = tuple(X[CLAIM_KEY_COL]) if CLAIM_KEY_COL in X.columns else None

    return HealthShapExplanation(
        feature_names=tuple(feature_names),
        shap_values=shap_values,
        base_value=base_value,
        predicted_proba=predicted_proba,
        claim_keys=claim_keys,
    )


def explain_single_health_claim(model: HealthFraudModel, X_row: pd.DataFrame) -> dict:
    """Local explanation for exactly one claim, as a plain UI-ready dict.

    `X_row` must be a one-row DataFrame with the same columns
    `explain_health_claims` expects. Returns:
        {
            "claim_key": str or None,
            "predicted_proba": float,
            "base_value": float,
            "contributions": [{"feature": ..., "feature_value": ..., "shap_value": ...}, ...],
        }
    `contributions` is sorted by descending `|shap_value|` (strongest
    factors first).
    """
    if not isinstance(X_row, pd.DataFrame) or len(X_row) != 1:
        n = len(X_row) if isinstance(X_row, pd.DataFrame) else "N/A"
        raise HealthExplainabilityError(f"X_row must contain exactly one row, got {n}")

    feature_names = _require_fitted(model)
    explainer = _explainer_for(model)
    claim_key = X_row.iloc[0][CLAIM_KEY_COL] if CLAIM_KEY_COL in X_row.columns else None

    try:
        local = explain_one(explainer, X_row, feature_names, claim_key=claim_key)
    except ExplainabilityError as exc:
        raise HealthExplainabilityError(str(exc)) from exc

    predicted_proba = float(model.predict_proba(X_row)[0])
    contributions = [
        {
            "feature": name,
            "feature_value": local.feature_values[name],
            "shap_value": local.feature_contributions[name],
        }
        for name in local.ranked_features
    ]
    return {
        "claim_key": local.claim_key,
        "predicted_proba": predicted_proba,
        "base_value": local.base_value,
        "contributions": contributions,
    }
