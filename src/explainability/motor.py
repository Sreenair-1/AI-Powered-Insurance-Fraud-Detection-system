"""SHAP-based explainability for the fitted motor fraud risk model.

Wraps a fitted `MotorFraudModel` (`src.models.motor`) with the shared
model-agnostic SHAP layer (`src.explainability.common`) to produce global
feature importance and per-claim local explanations. This module never
fits or refits a model — it only reads a model already trained via
`MotorFraudModel.fit()`.

SHAP values explain *model behavior* (which features pushed a given
prediction up or down), never a fraud determination. A claim with a large positive SHAP total is
"elevated risk" per the model, not proven fraudulent.

SHAP values here are in the model's raw margin (log-odds) space, matching
`shap.TreeExplainer`'s default output for an XGBoost binary classifier —
`sigmoid(base_value + shap_values[row].sum())` reproduces
`model.predict_proba(X)[row]` (verified in `tests/test_shap_motor.py`).

Reconciliation with `src.explainability.common`: `explain_motor_claims`
still prepares `X` via `MotorFraudModel._prepare_inference_features()`
rather than `common.prepare_features_for_shap` — that private helper is
the only place the fit-time categorical dtype set (unseen category ->
`NaN`) is re-applied, and `common.py` deliberately knows nothing about
either model wrapper's categorical handling (see `docs/shap.md`). Once
`X` is prepared, explainer construction and SHAP value/base-value
normalization are delegated to `common.create_tree_explainer` /
`common.explain_local` instead of duplicating that logic here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from src.explainability import common
from src.features.structured import CLAIM_KEY_COL
from src.models.motor import MotorFraudModel, MotorModelError


@dataclass(frozen=True)
class MotorShapExplanation:
    """SHAP explanation for a batch of motor claims.

    Attributes:
        shap_values: `(n_rows, n_features)` array, row order matches the
            input `X`, column order matches `feature_names`.
        base_value: the explainer's expected model output (float, raw
            margin/log-odds space) that `shap_values` are offset from.
        feature_names: the fitted model's feature column order (excludes
            `claim_key`), as returned by `MotorFraudModel.feature_columns`.
        claim_keys: `claim_key` per row, same order as `shap_values` rows.
    """

    shap_values: np.ndarray
    base_value: float
    feature_names: list[str]
    claim_keys: list[str]

    def to_frame(self) -> pd.DataFrame:
        """SHAP values as a DataFrame: one row per claim, one column per feature."""
        df = pd.DataFrame(self.shap_values, columns=self.feature_names)
        df.insert(0, CLAIM_KEY_COL, self.claim_keys)
        return df

    def global_importance(self) -> pd.Series:
        """Mean `|SHAP value|` per feature across all rows, descending.

        A single-model, single-batch summary of which features drive
        predictions most on average — not a per-claim explanation.
        """
        importance = np.abs(self.shap_values).mean(axis=0)
        return pd.Series(importance, index=self.feature_names, name="mean_abs_shap").sort_values(
            ascending=False
        )

    def local_explanation(self, row: int) -> pd.Series:
        """Per-claim SHAP values for one row, ordered by `|value|` descending.

        The returned `pandas.Series` (index = feature name, value = signed
        SHAP contribution in margin/log-odds space) is the "strongest
        factors".

        Raises:
            MotorModelError: if `row` is outside `[0, n_rows)`.
        """
        n_rows = self.shap_values.shape[0]
        if not 0 <= row < n_rows:
            raise MotorModelError(f"row index {row} out of range for {n_rows} row(s)")
        values = pd.Series(self.shap_values[row], index=self.feature_names)
        return values.reindex(values.abs().sort_values(ascending=False).index)

    def summary_records(self, top_n: int = 10) -> list[dict[str, object]]:
        """One JSON-serializable dict per claim with its `top_n` strongest features.

        Intended as a ready-to-consume shape for a later Streamlit UI without
        that UI needing to know about `pandas`/`numpy` objects directly.

        Raises:
            MotorModelError: if `top_n` is not a positive integer.
        """
        if top_n <= 0:
            raise MotorModelError(f"top_n must be a positive integer, got {top_n}")

        records: list[dict[str, object]] = []
        for i in range(self.shap_values.shape[0]):
            local = self.local_explanation(i).head(top_n)
            records.append(
                {
                    CLAIM_KEY_COL: self.claim_keys[i],
                    "base_value": self.base_value,
                    "top_features": [
                        {"feature": name, "shap_value": float(value)}
                        for name, value in local.items()
                    ],
                }
            )
        return records


def explain_motor_claims(
    model: MotorFraudModel, X: Union[pd.DataFrame, pd.Series]
) -> MotorShapExplanation:
    """Compute SHAP values for `X` using an already-fitted `MotorFraudModel`.

    Never fits, refits, or otherwise mutates `model`. `X` is prepared
    exactly as `MotorFraudModel.predict_proba` prepares it internally (same
    fitted feature order, same fit-time categorical dtype set re-applied to
    unseen categories -> `NaN`) so SHAP values reflect what the model
    actually sees at inference time. Column reordering/validation and
    `claim_key` handling are therefore governed entirely by
    `MotorFraudModel`'s own feature contract, not reimplemented here.
    Explainer construction and SHAP value/base-value normalization are
    delegated to `src.explainability.common` (`create_tree_explainer`,
    `explain_local`) rather than duplicated in this module.

    Args:
        model: a `MotorFraudModel` that has already been `.fit()`.
        X: claim-keyed feature matrix (must contain `claim_key` plus every
            feature column the model was fit on — the same contract
            `model.predict`/`model.predict_proba` require).

    Raises:
        MotorModelError: if `model` has not been fit yet, or `X` fails the
            same validation `MotorFraudModel` applies at predict time
            (not a DataFrame, empty, missing `claim_key`, missing/extra
            feature columns).
    """
    claim_keys = X[CLAIM_KEY_COL].astype(str).tolist() if isinstance(X, pd.DataFrame) and CLAIM_KEY_COL in X.columns else None

    # `_prepare_inference_features` is the same private helper
    # `MotorFraudModel.predict`/`predict_proba` use internally — reused here
    # (not reimplemented) so SHAP input matches prediction input exactly,
    # including categorical dtype re-application. It performs its own
    # `claim_key`/column validation and raises `MotorModelError` on failure,
    # so no duplicate validation is done here first.
    features = model._prepare_inference_features(X)  # noqa: SLF001

    if claim_keys is None:
        # `_prepare_inference_features` succeeding guarantees `claim_key`
        # was present on `X` (it validates this itself), so this branch is
        # unreachable in practice; kept only as an explicit invariant guard
        # rather than a silent `None` fallback.
        raise MotorModelError(f"X must contain a '{CLAIM_KEY_COL}' column")

    feature_names = list(features.columns)
    explainer = common.create_tree_explainer(model._model)  # noqa: SLF001
    # `features` already carries the fit-time categorical dtypes re-applied
    # by `_prepare_inference_features` and is already ordered/filtered onto
    # `feature_names`, so `common.explain_local`'s own alignment step
    # (drop claim_key/leakage columns, reindex) is a no-op here — it just
    # supplies the shared, tested SHAP-value/base-value normalization.
    local_explanations = common.explain_local(
        explainer, features, feature_names, claim_keys=claim_keys
    )
    shap_values = np.array(
        [[le.feature_contributions[name] for name in feature_names] for le in local_explanations]
    )
    base_value = local_explanations[0].base_value

    return MotorShapExplanation(
        shap_values=shap_values,
        base_value=base_value,
        feature_names=feature_names,
        claim_keys=claim_keys,
    )
