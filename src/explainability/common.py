"""Shared SHAP explainability layer for fraud risk models.

Reusable across `HealthFraudModel` and `MotorFraudModel` (`src.models.health`
/ `src.models.motor`): both wrap a fitted `xgboost.XGBClassifier`, so this
module works directly against that fitted estimator rather than either
wrapper class, and never imports `src.models.health` / `src.models.motor` —
callers pass in the fitted estimator (e.g. `health_model.model` or
`motor_model._model`) plus the exact feature matrix/feature names the model
was fit on.

SHAP values here are computed with `shap.TreeExplainer` in its default
(tree-path-dependent, no background data required) mode. For an
XGBoost binary classifier this means: values are in **raw margin
(log-odds) space**, not probability space, and for every row
`base_value + sum(shap_values) == raw_margin_prediction`
(`model.predict(X, output_margin=True)`), not the predicted probability.
This is a standard SHAP/XGBoost convention — documented in `docs/shap.md`.

This module never fits or retrains a model, and never accepts or reads a
target/label column — `_LEAKAGE_COLUMNS` is stripped defensively from any
input feature matrix before an explanation is computed, matching the same
constant used in `src.features.fusion` / `src.features.structured`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import shap

from src.data.schemas import HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL, MOTOR_TARGET
from src.features.structured import CLAIM_KEY_COL

# Columns that must never be treated as model features / appear in an
# explanation, regardless of whether the caller's `X` still carries them —
# same set `src.features.fusion` / `src.features.structured` strip from a
# fused feature matrix.
_LEAKAGE_COLUMNS = frozenset(
    {
        MOTOR_TARGET,
        HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL,
        "synthetic_fraud_ring_label",
        "synthetic_ring_id",
    }
)

FeatureValue = Union[float, int, bool, str, None]


class ExplainabilityError(ValueError):
    """Raised for invalid inputs to the SHAP explainability layer."""


@dataclass(frozen=True)
class GlobalExplanation:
    """Dataset-level SHAP feature importance for one fitted model.

    `mean_abs_shap` is the mean absolute SHAP value per feature (raw margin
    space) over the rows the explanation was computed on — a standard global
    importance summary, not a claim-level result.
    """

    feature_names: list[str]
    mean_abs_shap: dict[str, float]
    ranked_feature_names: list[str]
    base_value: float
    n_samples: int

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """The `n` highest-importance features as `(name, mean_abs_shap)` pairs."""
        return [(name, self.mean_abs_shap[name]) for name in self.ranked_feature_names[:n]]

    def to_dict(self) -> dict[str, Any]:
        """Flat dict representation, e.g. for a Streamlit table or JSON logging."""
        return {
            "feature_names": list(self.feature_names),
            "mean_abs_shap": dict(self.mean_abs_shap),
            "ranked_feature_names": list(self.ranked_feature_names),
            "base_value": self.base_value,
            "n_samples": self.n_samples,
        }


@dataclass(frozen=True)
class LocalExplanation:
    """Per-claim SHAP explanation: one row's feature contributions to the model's prediction.

    `feature_contributions` and `base_value` are in raw margin (log-odds)
    space: `base_value + sum(feature_contributions.values())` reconstructs
    the model's raw margin output for this row (see module docstring), not
    the predicted probability. This explains *model behavior* for this row —
    it is not evidence of fraud on its own.
    """

    claim_key: Optional[str]
    base_value: float
    predicted_margin: float
    feature_contributions: dict[str, float]
    feature_values: dict[str, FeatureValue]
    ranked_features: list[str]

    def top_contributors(self, n: int = 10) -> list[tuple[str, float]]:
        """The `n` largest-magnitude contributors as `(name, shap_value)` pairs."""
        return [(name, self.feature_contributions[name]) for name in self.ranked_features[:n]]

    def to_dict(self) -> dict[str, Any]:
        """Flat dict representation, e.g. for a Streamlit table or JSON logging."""
        return {
            "claim_key": self.claim_key,
            "base_value": self.base_value,
            "predicted_margin": self.predicted_margin,
            "feature_contributions": dict(self.feature_contributions),
            "feature_values": dict(self.feature_values),
            "ranked_features": list(self.ranked_features),
        }


def _validate_fitted_estimator(estimator: Any) -> None:
    """Raise `ExplainabilityError` unless `estimator` is a fitted, probability-capable model."""
    if estimator is None:
        raise ExplainabilityError(
            "estimator is None; the model must be fit before creating a SHAP explainer"
        )
    if not hasattr(estimator, "predict_proba"):
        raise ExplainabilityError(
            f"estimator of type {type(estimator).__name__!r} has no predict_proba(); "
            "SHAP explanations require a fitted probabilistic classifier"
        )
    get_booster = getattr(estimator, "get_booster", None)
    if callable(get_booster):
        try:
            get_booster()
        except Exception as exc:
            raise ExplainabilityError(
                "estimator is not fitted yet (no booster available); call .fit() first"
            ) from exc


def create_tree_explainer(estimator: Any) -> shap.TreeExplainer:
    """Build a `shap.TreeExplainer` for a fitted tree-based estimator.

    Args:
        estimator: A fitted `xgboost.XGBClassifier` (or any estimator
            `shap.TreeExplainer` supports) — e.g. `HealthFraudModel.model`
            or `MotorFraudModel`'s fitted underlying `XGBClassifier`. Not the
            `HealthFraudModel`/`MotorFraudModel` wrapper itself.

    Raises:
        ExplainabilityError: if `estimator` is `None`, unfitted, or has no
            `predict_proba`.
    """
    _validate_fitted_estimator(estimator)
    return shap.TreeExplainer(estimator)


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in _LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=present) if present else df


def prepare_features_for_shap(X: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Align `X` onto the exact `feature_names` (and order) a model was fit on.

    Drops `claim_key` (a row identifier, never a feature) and any known
    leakage/target column if present, then reindexes columns onto
    `feature_names` in that exact order — the same alignment contract
    `HealthFraudModel`/`MotorFraudModel` apply internally before calling the
    underlying XGBoost estimator, so SHAP values line up with the columns
    the model actually saw.

    Args:
        X: Feature matrix, e.g. `fuse_health_claims()`/`fuse_motor_claims()`
            output. May still carry `claim_key`.
        feature_names: The exact feature columns (and order) the fitted
            estimator was trained on (e.g. `HealthFraudModel.feature_names_`
            or `MotorFraudModel.feature_columns`).

    Raises:
        ExplainabilityError: if `X` is not a non-empty `DataFrame`, or is
            missing any column in `feature_names`.
    """
    if not isinstance(X, pd.DataFrame):
        raise ExplainabilityError(f"X must be a pandas DataFrame, got {type(X).__name__!r}")
    if X.empty:
        raise ExplainabilityError("X has no rows")
    feature_names = list(feature_names)
    if not feature_names:
        raise ExplainabilityError("feature_names is empty")

    features = X.drop(columns=[CLAIM_KEY_COL]) if CLAIM_KEY_COL in X.columns else X
    features = _drop_leakage_columns(features)

    missing = set(feature_names) - set(features.columns)
    if missing:
        raise ExplainabilityError(
            f"X is missing feature column(s) required by the fitted model: {sorted(missing)}"
        )
    return features[feature_names]


def _shap_values_2d(explainer: shap.TreeExplainer, features: pd.DataFrame) -> np.ndarray:
    """Return SHAP values as a plain `(n_samples, n_features)` array.

    `shap.TreeExplainer.shap_values()` returns a single 2D array for an
    `XGBClassifier` binary model (verified against the installed shap/xgboost
    versions); older shap releases can return a length-2 list (one array per
    class) for other binary classifiers, so that shape is also handled here
    defensively, taking the positive-class (index 1) array.
    """
    raw = explainer.shap_values(features)
    if isinstance(raw, list):
        if len(raw) != 2:
            raise ExplainabilityError(
                f"expected shap_values() to return 2 classes for a binary classifier, got {len(raw)}"
            )
        raw = raw[1]
    array = np.asarray(raw)
    if array.ndim != 2:
        raise ExplainabilityError(
            f"expected 2D SHAP values (n_samples, n_features), got shape {array.shape}"
        )
    return array


def _scalar_base_value(expected_value: Any) -> float:
    """Normalize `TreeExplainer.expected_value` (scalar, or length-1/2 array/list) to a float."""
    value = np.asarray(expected_value).reshape(-1)
    if value.size == 0:
        raise ExplainabilityError("explainer.expected_value is empty")
    # A length-2 expected_value (one per class) pairs with the shap_values()
    # list-of-two-arrays case in `_shap_values_2d` — take the positive class.
    return float(value[-1])


def _to_native(value: Any) -> FeatureValue:
    """Convert a pandas/numpy scalar to a plain Python value for display/serialization."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def explain_global(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    feature_names: Sequence[str],
) -> GlobalExplanation:
    """Compute dataset-level SHAP feature importance for `X`.

    Args:
        explainer: From `create_tree_explainer()`.
        X: Feature matrix to explain (e.g. a validation split). May still
            carry `claim_key`.
        feature_names: The exact feature columns the fitted estimator was
            trained on, in fit order.

    Raises:
        ExplainabilityError: see `prepare_features_for_shap`.
    """
    features = prepare_features_for_shap(X, feature_names)
    feature_names = list(feature_names)

    shap_values = _shap_values_2d(explainer, features)
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_abs_shap = {name: float(val) for name, val in zip(feature_names, mean_abs)}
    ranked = sorted(feature_names, key=lambda name: mean_abs_shap[name], reverse=True)

    return GlobalExplanation(
        feature_names=feature_names,
        mean_abs_shap=mean_abs_shap,
        ranked_feature_names=ranked,
        base_value=_scalar_base_value(explainer.expected_value),
        n_samples=len(features),
    )


def explain_local(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    feature_names: Sequence[str],
    claim_keys: Optional[Sequence[Optional[str]]] = None,
) -> list[LocalExplanation]:
    """Compute one per-row (per-claim) SHAP explanation for every row of `X`.

    Args:
        explainer: From `create_tree_explainer()`.
        X: Feature matrix to explain, one row per claim. May still carry
            `claim_key`.
        feature_names: The exact feature columns the fitted estimator was
            trained on, in fit order.
        claim_keys: Optional explicit claim identifiers, same length/order
            as `X`. If omitted, `X`'s own `claim_key` column is used when
            present; otherwise every `LocalExplanation.claim_key` is `None`.

    Raises:
        ExplainabilityError: see `prepare_features_for_shap`, or if
            `claim_keys` is given with a length that doesn't match `X`.
    """
    features = prepare_features_for_shap(X, feature_names)
    feature_names = list(feature_names)

    if claim_keys is not None:
        claim_keys = list(claim_keys)
        if len(claim_keys) != len(features):
            raise ExplainabilityError(
                f"claim_keys has {len(claim_keys)} entries but X has {len(features)} rows"
            )
    elif CLAIM_KEY_COL in X.columns:
        claim_keys = list(X.loc[features.index, CLAIM_KEY_COL])
    else:
        claim_keys = [None] * len(features)

    shap_values = _shap_values_2d(explainer, features)
    base_value = _scalar_base_value(explainer.expected_value)

    explanations: list[LocalExplanation] = []
    for i in range(len(features)):
        row = features.iloc[i]
        contributions = {name: float(shap_values[i, j]) for j, name in enumerate(feature_names)}
        ranked = sorted(feature_names, key=lambda name: abs(contributions[name]), reverse=True)
        explanations.append(
            LocalExplanation(
                claim_key=claim_keys[i],
                base_value=base_value,
                predicted_margin=float(base_value + shap_values[i].sum()),
                feature_contributions=contributions,
                feature_values={name: _to_native(row[name]) for name in feature_names},
                ranked_features=ranked,
            )
        )
    return explanations


def explain_one(
    explainer: shap.TreeExplainer,
    x_row: pd.DataFrame,
    feature_names: Sequence[str],
    claim_key: Optional[str] = None,
) -> LocalExplanation:
    """Convenience wrapper: explain a single-row `DataFrame` (one claim).

    Raises:
        ExplainabilityError: if `x_row` does not have exactly one row, or
            see `prepare_features_for_shap`.
    """
    if not isinstance(x_row, pd.DataFrame) or len(x_row) != 1:
        n = len(x_row) if isinstance(x_row, pd.DataFrame) else "N/A"
        raise ExplainabilityError(f"x_row must be a single-row DataFrame, got {n} row(s)")
    keys = [claim_key] if claim_key is not None else None
    return explain_local(explainer, x_row, feature_names, claim_keys=keys)[0]
