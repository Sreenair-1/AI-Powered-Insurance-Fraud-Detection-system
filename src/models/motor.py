
"""Motor fraud risk model — XGBoost classifier.

Consumes src.features.fusion.fuse_motor_claims() output (X, y).
The claim_key column is dropped internally and never used as a feature.
The FraudFound_P target is supplied separately as a pandas Series.

Class imbalance is handled using scale_pos_weight, calculated from the
training target at fit time.

This model provides risk scores for decision support; it does not prove fraud.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features.structured import CLAIM_KEY_COL

RANDOM_STATE = 42


# Features that may exist in the current inference pipeline but may not
# have been included when the saved model was trained.
# These columns are ignored only when they are not part of the trained schema.
_OPTIONAL_INFERENCE_FEATURES = frozenset({
    # Synthetic graph features
    "GarageID",
    "BankAccountID",
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "community_id",
    "num_entities",
    "max_entity_claim_degree",
    "mean_entity_claim_degree",
    "has_shared_entity",

    # Copy-move document-forensics features
    "copy_move_available",
    "copy_move_match_ratio",
    "copy_move_max_cluster_size",
    "copy_move_mean_match_distance",
    "copy_move_num_filtered_matches",
    "copy_move_num_keypoints",
    "copy_move_num_raw_matches",
    "copy_move_potential_duplicate_regions_found",
    "copy_move_std_match_distance",

    # Error Level Analysis (ELA) features
    "ela_available",
    "ela_high_error_pixel_ratio",
    "ela_max_error",
    "ela_mean_error",
    "ela_median_error",
    "ela_p95_error",
    "ela_std_error",

    # File and image metadata features
    "metadata_aspect_ratio",
    "metadata_available",
    "metadata_datetime_original_modified_mismatch",
    "metadata_exif_field_count",
    "metadata_exif_malformed",
    "metadata_file_size_bytes",
    "metadata_has_camera_info",
    "metadata_has_datetime_modified",
    "metadata_has_datetime_original",
    "metadata_has_exif",
    "metadata_has_software_tag",
    "metadata_height",
    "metadata_software_editing_indicator",
    "metadata_width",
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
    """Validate X and drop claim_key, which is a join key, not a feature."""
    if not isinstance(X, pd.DataFrame):
        raise MotorModelError(
            f"X must be a pandas DataFrame, got {type(X)!r}"
        )

    if X.empty:
        raise MotorModelError("X has no rows")

    if CLAIM_KEY_COL not in X.columns:
        raise MotorModelError(
            f"X must contain a '{CLAIM_KEY_COL}' column "
            "(use src.features.fusion.fuse_motor_claims output)"
        )

    return X.drop(columns=[CLAIM_KEY_COL])


def _validate_target(y: pd.Series, n_rows: int) -> pd.Series:
    """Validate the binary training target."""
    if not isinstance(y, pd.Series):
        raise MotorModelError(
            f"y must be a pandas Series, got {type(y)!r}"
        )

    if len(y) != n_rows:
        raise MotorModelError(
            f"y has {len(y)} rows but X has {n_rows} rows"
        )

    if y.isna().any():
        raise MotorModelError("y contains missing values")

    unique = set(pd.unique(y).tolist())

    if not unique <= {0, 1}:
        raise MotorModelError(
            f"y must be binary (0/1); got values {sorted(unique)}"
        )

    if len(unique) < 2:
        raise MotorModelError(
            "y must contain both classes (0 and 1) to fit a classifier; "
            f"got only {sorted(unique)}"
        )

    return y.astype(int)


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Calculate the negative-to-positive class ratio."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / pos


# Dtype kinds accepted natively by XGBoost's categorical training path.
_NATIVE_DTYPE_KINDS = frozenset({"i", "u", "f", "b"})


def _coerce_object_columns_to_category(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Convert non-numeric, non-boolean columns to categorical dtype."""
    features = features.copy()

    for col in features.columns:
        dtype = features[col].dtype

        if (
            isinstance(dtype, pd.CategoricalDtype)
            or dtype.kind in _NATIVE_DTYPE_KINDS
        ):
            continue

        features[col] = features[col].astype("category")

    return features


def _categorical_dtypes(
    features: pd.DataFrame,
) -> dict[str, pd.CategoricalDtype]:
    """Record categorical dtypes observed during training."""
    return {
        str(col): dtype
        for col, dtype in features.dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype)
    }


def _apply_categorical_dtypes(
    features: pd.DataFrame,
    dtypes: dict[str, pd.CategoricalDtype],
) -> pd.DataFrame:
    """Restore the exact categorical values and ordering used during fit."""
    features = features.copy()

    for col, dtype in dtypes.items():
        raw = features[col]
        known = raw.isin(dtype.categories)

        # Values not observed during training become missing.
        masked = raw.where(known).astype(object)

        features[col] = pd.Categorical(
            masked,
            categories=dtype.categories,
            ordered=dtype.ordered,
        )

    return features


class MotorFraudModel:
    """XGBoost risk classifier for motor insurance claims."""

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params: dict = {**DEFAULT_XGB_PARAMS, **(params or {})}
        self._model: Optional[XGBClassifier] = None
        self._feature_columns: Optional[list[str]] = None
        self._categorical_dtypes: dict[
            str, pd.CategoricalDtype
        ] = {}
        self._scale_pos_weight: Optional[float] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "MotorFraudModel":
        """Train on the training split only."""
        features = _validate_features(X)
        features = _coerce_object_columns_to_category(features)
        target = _validate_target(y, len(features))

        self._feature_columns = list(features.columns)
        self._categorical_dtypes = _categorical_dtypes(features)
        self._scale_pos_weight = _compute_scale_pos_weight(target)

        model = XGBClassifier(
            scale_pos_weight=self._scale_pos_weight,
            **self.params,
        )

        model.fit(features, target)
        self._model = model

        return self

    def _prepare_inference_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate inference columns and align them to the trained schema."""
        if self._model is None or self._feature_columns is None:
            raise MotorModelError(
                "model has not been fit yet; call fit() first"
            )

        features = _validate_features(X)

        # Every feature required by the trained model must be present.
        missing = set(self._feature_columns) - set(features.columns)

        if missing:
            raise MotorModelError(
                "input is missing feature columns seen at fit time: "
                f"{sorted(missing)}"
            )

        # Permit only explicitly known optional columns beyond the model schema.
        extra = set(features.columns) - set(self._feature_columns)
        unexpected = extra - _OPTIONAL_INFERENCE_FEATURES

        if unexpected:
            raise MotorModelError(
                "input contains unexpected feature column(s): "
                f"{sorted(unexpected)}"
            )

        # Remove extra columns and preserve the model's exact feature order.
        # Optional graph/forensics columns are ignored if not used during fit.
        features = features.reindex(
            columns=self._feature_columns
        ).copy()

        features = _apply_categorical_dtypes(
            features,
            self._categorical_dtypes,
        )

        return features

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted probability of class 1 for each claim."""
        features = self._prepare_inference_features(X)

        model = self._model
        assert model is not None

        return model.predict_proba(features)[:, 1]

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary risk flags using the supplied probability threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise MotorModelError(
                f"threshold must be in [0, 1], got {threshold}"
            )

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @property
    def scale_pos_weight(self) -> Optional[float]:
        """Return the class-imbalance weight calculated during training."""
        return self._scale_pos_weight

    @property
    def feature_columns(self) -> Optional[list[str]]:
        """Return the trained feature names, excluding claim_key."""
        return (
            list(self._feature_columns)
            if self._feature_columns is not None
            else None
        )
