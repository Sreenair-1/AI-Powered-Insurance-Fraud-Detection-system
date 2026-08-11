"""SHAP explainability for the health fraud risk model.

Wraps `shap.TreeExplainer` around an already-fitted
`src.models.health.HealthFraudModel` to produce global feature
importance and per-claim local explanations. Never retrains or refits
the model — every function here only reads `model.model` (the fitted
`XGBClassifier`) and `model.feature_names_`.

SHAP values are computed in the model's native margin (log-odds) output
space, so `base_value + sum(shap_values for a row) == raw model margin`
for that row (the standard additive SHAP guarantee for
`shap.TreeExplainer` on a tree-ensemble classifier). Predicted
probabilities are reported separately (via `model.predict_proba`) for
UI display, not derived from the SHAP values themselves.

Feature alignment mirrors `HealthFraudModel._prepare` exactly: `claim_key`
is dropped (row identifier, not a feature) and the remaining columns are
reindexed onto `model.feature_names_`, the exact order fixed at fit time.
This also guarantees `claim_fraud_label_provider_derived` can never reach
SHAP as a feature, since `HealthFraudModel.fit()` never includes it in
`feature_names_` in the first place, and any extra column on the input
`X` that isn't in `feature_names_` is simply dropped here.

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
import shap

from src.features.structured import CLAIM_KEY_COL
from src.data.schemas import HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL
from src.models.health import HealthFraudModel


class HealthExplainabilityError(ValueError):
    """Raised for invalid input/state when explaining a HealthFraudModel."""


def _prepare_features(model: HealthFraudModel, X: pd.DataFrame) -> pd.DataFrame:
    """Reindex `X` onto `model.feature_names_`, preserving fitted feature order.

    Mirrors `HealthFraudModel._prepare` exactly (duplicated rather than
    calling that private method directly, so this module doesn't depend on
    another module's private implementation detail) — same drop-`claim_key`
    then reindex-onto-`feature_names_` behavior, same errors.
    """
    if model.model is None or model.feature_names_ is None:
        raise HealthExplainabilityError(
            "HealthFraudModel must be fit() before it can be explained"
        )
    features = X.drop(columns=[CLAIM_KEY_COL]) if CLAIM_KEY_COL in X.columns else X
    missing = set(model.feature_names_) - set(features.columns)
    if missing:
        raise HealthExplainabilityError(
            f"X is missing feature column(s) seen at fit time: {sorted(missing)}"
        )
    features = features[model.feature_names_]
    if HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL in features.columns:
        # Cannot happen via the reindex above (fit() never puts the label in
        # feature_names_), but guarded explicitly since this rule is safety-
        # critical: the label must never reach SHAP as a feature.
        raise HealthExplainabilityError(
            f"{HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL!r} must never be used as a feature"
        )
    return features


def _normalize_shap_output(raw_shap_values: object, expected_value: object) -> tuple[np.ndarray, float]:
    """Normalize `TreeExplainer` output across shap's binary-classifier shapes.

    Different shap versions have returned binary-classification output as
    either a single `(n_samples, n_features)` array (installed version,
    shap 0.52) or a `[negative_class, positive_class]` list of arrays
    (older versions). Either way we want the positive-class contributions
    as one `(n_samples, n_features)` array and a matching scalar base
    value.
    """
    if isinstance(raw_shap_values, list):
        shap_values = np.asarray(raw_shap_values[-1])
    else:
        shap_values = np.asarray(raw_shap_values)
    if shap_values.ndim == 3:
        # (n_samples, n_features, n_outputs) -> take the positive class.
        shap_values = shap_values[:, :, -1]

    if isinstance(expected_value, (list, tuple, np.ndarray)):
        base_value = float(np.asarray(expected_value).reshape(-1)[-1])
    else:
        base_value = float(expected_value)

    return shap_values, base_value


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
    features = _prepare_features(model, X)
    assert model.model is not None  # narrowed by _prepare_features

    explainer = shap.TreeExplainer(model.model)
    raw_shap_values = explainer.shap_values(features)
    shap_values, base_value = _normalize_shap_output(raw_shap_values, explainer.expected_value)

    predicted_proba = model.predict_proba(X)
    claim_keys = tuple(X[CLAIM_KEY_COL]) if CLAIM_KEY_COL in X.columns else None

    return HealthShapExplanation(
        feature_names=tuple(model.feature_names_),
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
    if len(X_row) != 1:
        raise HealthExplainabilityError(f"X_row must contain exactly one row, got {len(X_row)}")

    explanation = explain_health_claims(model, X_row)
    features = _prepare_features(model, X_row)
    contributions_df = explanation.claim_contributions(0)
    contributions_df["feature_value"] = contributions_df["feature"].map(
        lambda f: features.iloc[0][f]
    )

    claim_key = explanation.claim_keys[0] if explanation.claim_keys is not None else None
    return {
        "claim_key": claim_key,
        "predicted_proba": float(explanation.predicted_proba[0]),
        "base_value": explanation.base_value,
        "contributions": [
            {
                "feature": row.feature,
                "feature_value": row.feature_value,
                "shap_value": float(row.shap_value),
            }
            for row in contributions_df.itertuples(index=False)
        ],
    }
