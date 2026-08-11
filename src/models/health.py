"""Health XGBoost claim-level fraud risk model.

Consumes `src.features.fusion.fuse_health_claims()` output (`X`, `y`)
directly: `X` is the claim-keyed fused feature matrix, `y` is the
provider-derived claim target (`claim_fraud_label_provider_derived`, see
`docs/data_contract.md` section 11) as a separate `pandas.Series` — the
target is never merged into `X` and this module never renames it to bare
`fraud`.

This is a decision-support risk score, not a fraud determination
: predictions should be read as "elevated risk" /
"flagged for investigation," never as proof of fraud.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features.structured import CLAIM_KEY_COL

RANDOM_STATE = 42

# Reference positive rate for the full labeled health-provider population
# `scale_pos_weight` itself is always computed from the *training split's*
# own class counts (see `compute_scale_pos_weight`), never hard-coded to
# this figure, so the model adapts to whatever imbalance a given train
# split actually has.
HEALTH_REFERENCE_POSITIVE_RATE = 0.0935

_DEFAULT_XGB_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute `neg_count / pos_count` from a training target only.

    Must be called with the training split's target, never validation/test
    — fitting the imbalance ratio on anything but train would leak split
    information into the model.

    Raises:
        ValueError: if `y` has no positive examples (ratio undefined).
    """
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        raise ValueError("cannot compute scale_pos_weight: training target has no positive examples")
    return neg / pos


def _drop_claim_key(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=[CLAIM_KEY_COL]) if CLAIM_KEY_COL in X.columns else X


class HealthFraudModel:
    """XGBoost fraud risk model for health claims.

    Trained on `fuse_health_claims()` output. `claim_key` is treated as a
    row identifier, not a feature, and is dropped internally before
    fitting/prediction — the target is already kept separate by
    `fuse_health_claims()` and is never touched here.
    """

    def __init__(self, scale_pos_weight: Optional[float] = None, **xgb_params: object) -> None:
        """Args:
        scale_pos_weight: If given, used as-is instead of computing it from
            the training target in `fit()`. Mainly for tests/experiments;
            normal training should leave this `None` so the ratio reflects
            the actual training split.
        **xgb_params: Overrides merged onto the module's default
            `XGBClassifier` parameters (e.g. `n_estimators`, `max_depth`).
        """
        self._explicit_scale_pos_weight = scale_pos_weight
        self._params = {**_DEFAULT_XGB_PARAMS, **xgb_params}
        self.model: Optional[XGBClassifier] = None
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "HealthFraudModel":
        """Fit on the training split only.

        Never pass validation or test data here — `predict`/`predict_proba`
        are the only supported way to run the fitted model on val/test.
        """
        features = _drop_claim_key(X_train)
        scale_pos_weight = (
            self._explicit_scale_pos_weight
            if self._explicit_scale_pos_weight is not None
            else compute_scale_pos_weight(y_train)
        )
        self.model = XGBClassifier(scale_pos_weight=scale_pos_weight, **self._params)
        self.model.fit(features, y_train)
        self.feature_names_ = list(features.columns)
        return self

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.feature_names_ is None:
            raise RuntimeError("HealthFraudModel.fit() must be called before predict/predict_proba")
        features = _drop_claim_key(X)
        missing = set(self.feature_names_) - set(features.columns)
        if missing:
            raise ValueError(f"X is missing feature column(s) seen at fit time: {sorted(missing)}")
        return features[self.feature_names_]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted class labels (0/1) for each row of `X`."""
        features = self._prepare(X)
        return self.model.predict(features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted fraud-risk probability (positive class) for each row of `X`."""
        features = self._prepare(X)
        return self.model.predict_proba(features)[:, 1]
