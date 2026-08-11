"""Forensic feature adapter.

Flattens the result objects produced by `src.forensics.ela.compute_ela`,
`src.forensics.copy_move.compute_copy_move`, and
`src.forensics.metadata.analyze_metadata` into a single flat, model-ready
mapping of feature name -> numeric/bool value, with explicit `ela_`,
`copy_move_`, and `metadata_` prefixes.

Any of the three forensic results may be unavailable (e.g. no document was
attached to a claim, or a particular check was skipped). Each extractor
accepts `None` for a missing result and still returns the full set of keys
for that source, so the output schema (`FORENSIC_FEATURE_NAMES`) is stable
regardless of which forensic checks actually ran. A companion
`..._available` flag lets a downstream model distinguish "genuinely False/
zero" from "this check did not run" (missing numeric values are `NaN`,
missing bool values are `False`).

This module makes no fraud decision and applies no fraud threshold - it only
reshapes existing forensic outputs into a flat numeric representation.
"""
from __future__ import annotations

from typing import Optional, Union

from src.forensics.copy_move import CopyMoveResult
from src.forensics.ela import ELAResult
from src.forensics.metadata import MetadataResult

FeatureValue = Union[float, bool]
FeatureDict = dict[str, FeatureValue]

_NAN = float("nan")

_ELA_NUMERIC_FIELDS: tuple[str, ...] = (
    "mean_error",
    "std_error",
    "median_error",
    "max_error",
    "p95_error",
    "high_error_pixel_ratio",
)

_COPY_MOVE_NUMERIC_FIELDS: tuple[str, ...] = (
    "num_keypoints",
    "num_raw_matches",
    "num_filtered_matches",
    "match_ratio",
    "mean_match_distance",
    "std_match_distance",
    "max_cluster_size",
)

_METADATA_NUMERIC_FIELDS: tuple[str, ...] = (
    "width",
    "height",
    "aspect_ratio",
    "exif_field_count",
)

_METADATA_BOOL_FIELDS: tuple[str, ...] = (
    "has_exif",
    "has_camera_info",
    "has_software_tag",
    "software_editing_indicator",
    "has_datetime_original",
    "has_datetime_modified",
    "datetime_original_modified_mismatch",
    "exif_malformed",
)


def extract_ela_features(result: Optional[ELAResult]) -> FeatureDict:
    """Flatten an `ELAResult` into `ela_`-prefixed model-ready features.

    Args:
        result: An `ELAResult` from `compute_ela`, or `None` if ELA was not
            run for this claim/document.

    Returns:
        A flat dict with `ela_available` (bool) plus one `ela_<field>` float
        per `ELAFeatures` field. All values are `NaN` when `result is None`.
    """
    features: FeatureDict = {"ela_available": result is not None}
    for name in _ELA_NUMERIC_FIELDS:
        value = getattr(result.features, name) if result is not None else None
        features[f"ela_{name}"] = _NAN if value is None else float(value)
    return features


def extract_copy_move_features(result: Optional[CopyMoveResult]) -> FeatureDict:
    """Flatten a `CopyMoveResult` into `copy_move_`-prefixed model-ready features.

    Args:
        result: A `CopyMoveResult` from `compute_copy_move`, or `None` if
            copy-move detection was not run for this claim/document.

    Returns:
        A flat dict with `copy_move_available` (bool),
        `copy_move_potential_duplicate_regions_found` (bool, `False` when
        unavailable), plus one `copy_move_<field>` float per
        `CopyMoveFeatures` field (`NaN` when `result is None`).
    """
    features: FeatureDict = {"copy_move_available": result is not None}
    features["copy_move_potential_duplicate_regions_found"] = (
        bool(result.potential_duplicate_regions_found) if result is not None else False
    )
    for name in _COPY_MOVE_NUMERIC_FIELDS:
        value = getattr(result.features, name) if result is not None else None
        features[f"copy_move_{name}"] = _NAN if value is None else float(value)
    return features


def extract_metadata_features(result: Optional[MetadataResult]) -> FeatureDict:
    """Flatten a `MetadataResult` into `metadata_`-prefixed model-ready features.

    Args:
        result: A `MetadataResult` from `analyze_metadata`, or `None` if
            metadata analysis was not run for this claim/document.

    Returns:
        A flat dict with `metadata_available` (bool), one `metadata_<field>`
        float per numeric `MetadataFeatures` field (`NaN` when unavailable),
        one `metadata_<field>` bool per boolean `MetadataFeatures` field
        (`False` when unavailable), and `metadata_file_size_bytes` (float,
        `NaN` when unavailable or when the source image has no known file
        size, e.g. an in-memory image).
    """
    features: FeatureDict = {"metadata_available": result is not None}
    for name in _METADATA_NUMERIC_FIELDS:
        value = getattr(result.features, name) if result is not None else None
        features[f"metadata_{name}"] = _NAN if value is None else float(value)
    for name in _METADATA_BOOL_FIELDS:
        value = getattr(result.features, name) if result is not None else None
        features[f"metadata_{name}"] = bool(value) if value is not None else False
    file_size = result.features.file_size_bytes if result is not None else None
    features["metadata_file_size_bytes"] = _NAN if file_size is None else float(file_size)
    return features


def extract_forensic_features(
    ela_result: Optional[ELAResult] = None,
    copy_move_result: Optional[CopyMoveResult] = None,
    metadata_result: Optional[MetadataResult] = None,
) -> FeatureDict:
    """Combine ELA, copy-move, and metadata results into one flat feature dict.

    Any argument may be `None` if that forensic check was not run/available
    for the claim/document being featurized; the corresponding `..._`
    features are still present in the output (as `NaN`/`False`), so the
    returned key set always equals `FORENSIC_FEATURE_NAMES`.

    Args:
        ela_result: Output of `compute_ela`, or `None`.
        copy_move_result: Output of `compute_copy_move`, or `None`.
        metadata_result: Output of `analyze_metadata`, or `None`.

    Returns:
        A flat dict mapping prefixed feature name -> float/bool value,
        suitable for assembling into a model feature matrix (e.g. one row
        of a `pandas.DataFrame`).
    """
    features: FeatureDict = {}
    features.update(extract_ela_features(ela_result))
    features.update(extract_copy_move_features(copy_move_result))
    features.update(extract_metadata_features(metadata_result))
    return features


FORENSIC_FEATURE_NAMES: tuple[str, ...] = tuple(extract_forensic_features().keys())
