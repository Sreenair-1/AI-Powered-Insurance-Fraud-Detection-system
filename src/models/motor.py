"""Motor fraud risk model — XGBoost classifier.

Consumes `src.features.fusion.fuse_motor_claims()` output (`X`, `y`)
directly: `X` is the claim-keyed feature matrix (must carry `claim_key`,
dropped internally, never used as a feature), `y` is the binary
`FraudFound_P` target as a separate `pandas.Series`. This module never
merges the target into the feature matrix and never derives it from `X`.

Class imbalance (motor fraud positive rate ~5.99%) is handled explicitly via
`scale_pos_weight = negative_count / positive_count`, computed from the
training target at fit time — not hard-coded, so it stays correct if the
training subset's balance changes.

This module produces "high-risk"/"suspicious" scores only, it is not a fraud adjudication.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features.structured import CLAIM_KEY_COL

RANDOM_STATE = 42

# These columns may be present in the current fused inference table even
# when a leakage-safe structured-only artifact was trained. Arbitrary new
# feature columns remain errors so schema mistakes are still caught.
_OPTIONAL_INFERENCE_FEATURES = frozenset({
    "GarageID", "BankAccountID", "degree_centrality", "betweenness_centrality",
    "closeness_centrality", "eigenvector_centrality", "community_id",
    "num_entities", "max_entity_claim_degree", "mean_entity_claim_degree",
    "has_shared_entity",
})

DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
}


class MotorModelError(ValueError):
    """Raised when inputs to the motor fraud model are invalid."""


def _validate_features(X: pd.DataFrame) -> pd.DataFrame:
    """Validate `X` and return it with `claim_key` dropped (join key, never a feature)."""
    if not isinstance(X, pd.DataFrame):
        raise MotorModelError(f"X must be a pandas DataFrame, got {type(X)!r}")
    if X.empty:
        raise MotorModelError("X has no rows")
    if CLAIM_KEY_COL not in X.columns:
        raise MotorModelError(
            f"X must contain a '{CLAIM_KEY_COL}' column (use src.features.fusion.fuse_motor_claims output)"
        )
    return X.drop(columns=[CLAIM_KEY_COL])


def _validate_target(y: pd.Series, n_rows: int) -> pd.Series:
    if not isinstance(y, pd.Series):
        raise MotorModelError(f"y must be a pandas Series, got {type(y)!r}")
    if len(y) != n_rows:
        raise MotorModelError(f"y has {len(y)} rows but X has {n_rows} rows")
    if y.isna().any():
        raise MotorModelError("y contains missing values")
    unique = set(pd.unique(y).tolist())
    if not unique <= {0, 1}:
        raise MotorModelError(f"y must be binary (0/1); got values {sorted(unique)}")
    if len(unique) < 2:
        raise MotorModelError(
            f"y must contain both classes (0 and 1) to fit a classifier; got only {sorted(unique)}"
        )
    return y.astype(int)


def _compute_scale_pos_weight(y: pd.Series) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / pos


# Dtype kinds XGBoost's enable_categorical path accepts natively without
# conversion: signed/unsigned int, float, bool. Everything else (plain
# `object`, pandas' `str` extension dtype, etc.) must become 'category'.
_NATIVE_DTYPE_KINDS = frozenset({"i", "u", "f", "b"})


def _coerce_object_columns_to_category(features: pd.DataFrame) -> pd.DataFrame:
    """Cast any non-numeric, non-bool, non-category columns to 'category' dtype.

    `fuse_motor_claims()` output can carry string-typed columns (e.g. the
    synthetic `GarageID`/`BankAccountID` entity ids from
    `src.data.synthetic`, which never pass through
    `src.data.preprocessing.encode_motor_categoricals`) that XGBoost's
    `enable_categorical` path does not accept as-is — including pandas'
    modern `str` extension dtype, which is not the classic `object` dtype
    but still needs this conversion.
    """
    features = features.copy()
    for col in features.columns:
        dtype = features[col].dtype
        if isinstance(dtype, pd.CategoricalDtype) or dtype.kind in _NATIVE_DTYPE_KINDS:
            continue
        features[col] = features[col].astype("category")
    return features


def _categorical_dtypes(features: pd.DataFrame) -> dict[str, pd.CategoricalDtype]:
    return {
        str(col): dtype
        for col, dtype in features.dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype)
    }


def _apply_categorical_dtypes(
    features: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype]
) -> pd.DataFrame:
    """Re-apply the exact fit-time category set to `features`.

    A category value present at inference time but absent from the fit-time
    category set becomes a missing value (`NaN`) rather than silently
    shifting another category's internal code — XGBoost's categorical split
    logic depends on codes being consistent between train and inference.
    Values are masked with `.where(...)` before construction so this never
    relies on pandas' deprecated "categorical dtype with out-of-category
    values" construction path.
    """
    features = features.copy()
    for col, dtype in dtypes.items():
        raw = features[col]
        known = raw.isin(dtype.categories)
        # Cast to plain object before rebuilding: a masked-but-still-categorical
        # Series keeps its old (superset) category metadata even after the
        # out-of-category values are replaced with NaN, which trips pandas'
        # deprecated-construction check below even though no actual value is
        # out of range. Object dtype carries no such metadata.
        masked = raw.where(known).astype(object)
        features[col] = pd.Categorical(masked, categories=dtype.categories, ordered=dtype.ordered)
    return features


class MotorFraudModel:
    """XGBoost fraud risk classifier for motor claims.

    Fit on the output of `fuse_motor_claims()`. Categorical columns (pandas
    `category` dtype, as produced by `src.features.structured`) are handled
    natively via XGBoost's `enable_categorical` support; the exact category
    set seen at fit time is recorded and re-applied to any data passed to
    `predict`/`predict_proba`, so a category unseen during training becomes
    a missing value rather than silently shifting another category's code.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params: dict = {**DEFAULT_XGB_PARAMS, **(params or {})}
        self._model: Optional[XGBClassifier] = None
        self._feature_columns: Optional[list[str]] = None
        self._categorical_dtypes: dict[str, pd.CategoricalDtype] = {}
        self._scale_pos_weight: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MotorFraudModel":
        """Train on `(X, y)` — typically `fuse_motor_claims()` output for the train split only.

        Never call this with validation/test data: the caller is responsible
        for only passing the train split (see `src.data.splits`).
        """
        features = _validate_features(X)
        features = _coerce_object_columns_to_category(features)
        target = _validate_target(y, len(features))

        self._feature_columns = list(features.columns)
        self._categorical_dtypes = _categorical_dtypes(features)
        self._scale_pos_weight = _compute_scale_pos_weight(target)

        model = XGBClassifier(scale_pos_weight=self._scale_pos_weight, **self.params)
        model.fit(features, target)
        self._model = model
        return self


    def _prepare_inference_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self._feature_columns is None:
            raise MotorModelError(
                "model has not been fit yet; call fit() first"
            )

        features = _validate_features(X)

        missing = set(self._feature_columns) - set(features.columns)
        if missing:
            raise MotorModelError(
                f"input is missing feature columns seen at fit time: "
                f"{sorted(missing)}"
            )

        extra = set(features.columns) - set(self._feature_columns)
        unexpected = extra - _OPTIONAL_INFERENCE_FEATURES
        if unexpected:
            raise MotorModelError(
                f"input contains unexpected feature column(s): "
                f"{sorted(unexpected)}"
            )

        # Known optional inference features may appear when the feature
        # pipeline is newer than a structured-only trained model. They are
        # deliberately ignored; arbitrary columns are rejected above.
        features = features.reindex(columns=self._feature_columns).copy()

        features = _apply_categorical_dtypes(
            features, self._categorical_dtypes
        )

        return features


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted fraud probability (class 1) for each row of `X`."""
        features = self._prepare_inference_features(X)
        model = self._model
        assert model is not None 
        return model.predict_proba(features)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary risk flags (1 = flagged high-risk) for each row of `X`."""
        if not 0.0 <= threshold <= 1.0:
            raise MotorModelError(f"threshold must be in [0, 1], got {threshold}")
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @property
    def scale_pos_weight(self) -> Optional[float]:
        """The `scale_pos_weight` (negative/positive count) computed at fit time, or `None` if unfit."""
        return self._scale_pos_weight

    @property
    def feature_columns(self) -> Optional[list[str]]:
        """The feature column names seen at fit time (excludes `claim_key`), or `None` if unfit."""
        return list(self._feature_columns) if self._feature_columns is not None else None
