"""Unit tests for src/features/forensic.py, using synthetic in-memory
forensic result objects only (no images, no data/raw/ dependency)."""
from __future__ import annotations

import copy
import math

from PIL import Image

from src.features.forensic import (
    FORENSIC_FEATURE_NAMES,
    extract_copy_move_features,
    extract_ela_features,
    extract_forensic_features,
    extract_metadata_features,
)
from src.forensics.copy_move import CopyMoveFeatures, CopyMoveResult
from src.forensics.ela import ELAFeatures, ELAResult
from src.forensics.metadata import MetadataFeatures, MetadataResult


def _make_ela_result() -> ELAResult:
    features = ELAFeatures(
        mean_error=1.5,
        std_error=2.5,
        median_error=1.0,
        max_error=40.0,
        p95_error=10.0,
        high_error_pixel_ratio=0.05,
    )
    return ELAResult(
        ela_image=Image.new("RGB", (4, 4)),
        features=features,
        metadata={"original_format": "JPEG", "original_size": (4, 4), "quality": 90, "scale": 10.0},
    )


def _make_copy_move_result(found: bool = True) -> CopyMoveResult:
    features = CopyMoveFeatures(
        num_keypoints=100,
        num_raw_matches=10,
        num_filtered_matches=3 if found else 0,
        match_ratio=0.03 if found else 0.0,
        mean_match_distance=25.0,
        std_match_distance=5.0,
        max_cluster_size=2,
    )
    return CopyMoveResult(
        potential_duplicate_regions_found=found,
        features=features,
        visualization=None,
        metadata={"original_format": "PNG"},
    )


def _make_metadata_result(file_size: int | None = 12345) -> MetadataResult:
    features = MetadataFeatures(
        width=200,
        height=100,
        aspect_ratio=2.0,
        file_size_bytes=file_size,
        has_exif=True,
        exif_field_count=4,
        has_camera_info=True,
        has_software_tag=True,
        software_editing_indicator=True,
        has_datetime_original=True,
        has_datetime_modified=True,
        datetime_original_modified_mismatch=True,
        exif_malformed=False,
    )
    return MetadataResult(
        structured_metadata={"format": "JPEG", "width": 200, "height": 100},
        features=features,
        analysis={"indicators": [], "note": "note"},
    )


# --- extract_ela_features ---------------------------------------------------


def test_ela_features_present_values():
    result = _make_ela_result()
    features = extract_ela_features(result)
    assert features["ela_available"] is True
    assert features["ela_mean_error"] == 1.5
    assert features["ela_std_error"] == 2.5
    assert features["ela_median_error"] == 1.0
    assert features["ela_max_error"] == 40.0
    assert features["ela_p95_error"] == 10.0
    assert features["ela_high_error_pixel_ratio"] == 0.05


def test_ela_features_missing_result_is_nan():
    features = extract_ela_features(None)
    assert features["ela_available"] is False
    for key, value in features.items():
        if key == "ela_available":
            continue
        assert math.isnan(value)


def test_ela_features_value_types_are_float():
    features = extract_ela_features(_make_ela_result())
    for key, value in features.items():
        if key == "ela_available":
            continue
        assert isinstance(value, float)


def test_ela_features_does_not_mutate_source():
    result = _make_ela_result()
    before = copy.deepcopy(result.features)
    extract_ela_features(result)
    assert result.features == before


# --- extract_copy_move_features ---------------------------------------------


def test_copy_move_features_present_values():
    result = _make_copy_move_result(found=True)
    features = extract_copy_move_features(result)
    assert features["copy_move_available"] is True
    assert features["copy_move_potential_duplicate_regions_found"] is True
    assert features["copy_move_num_keypoints"] == 100.0
    assert features["copy_move_num_raw_matches"] == 10.0
    assert features["copy_move_num_filtered_matches"] == 3.0
    assert features["copy_move_match_ratio"] == 0.03
    assert features["copy_move_mean_match_distance"] == 25.0
    assert features["copy_move_std_match_distance"] == 5.0
    assert features["copy_move_max_cluster_size"] == 2.0


def test_copy_move_features_no_duplicates_found():
    result = _make_copy_move_result(found=False)
    features = extract_copy_move_features(result)
    assert features["copy_move_available"] is True
    assert features["copy_move_potential_duplicate_regions_found"] is False
    assert features["copy_move_num_filtered_matches"] == 0.0


def test_copy_move_features_missing_result_is_safe_default():
    features = extract_copy_move_features(None)
    assert features["copy_move_available"] is False
    assert features["copy_move_potential_duplicate_regions_found"] is False
    for key, value in features.items():
        if key in ("copy_move_available", "copy_move_potential_duplicate_regions_found"):
            continue
        assert math.isnan(value)


def test_copy_move_features_does_not_mutate_source():
    result = _make_copy_move_result()
    before = copy.deepcopy(result.features)
    extract_copy_move_features(result)
    assert result.features == before


# --- extract_metadata_features -----------------------------------------------


def test_metadata_features_present_values():
    result = _make_metadata_result(file_size=12345)
    features = extract_metadata_features(result)
    assert features["metadata_available"] is True
    assert features["metadata_width"] == 200.0
    assert features["metadata_height"] == 100.0
    assert features["metadata_aspect_ratio"] == 2.0
    assert features["metadata_exif_field_count"] == 4.0
    assert features["metadata_file_size_bytes"] == 12345.0
    assert features["metadata_has_exif"] is True
    assert features["metadata_has_camera_info"] is True
    assert features["metadata_has_software_tag"] is True
    assert features["metadata_software_editing_indicator"] is True
    assert features["metadata_has_datetime_original"] is True
    assert features["metadata_has_datetime_modified"] is True
    assert features["metadata_datetime_original_modified_mismatch"] is True
    assert features["metadata_exif_malformed"] is False


def test_metadata_features_unknown_file_size_is_nan():
    result = _make_metadata_result(file_size=None)
    features = extract_metadata_features(result)
    assert features["metadata_available"] is True
    assert math.isnan(features["metadata_file_size_bytes"])


def test_metadata_features_missing_result_is_safe_default():
    features = extract_metadata_features(None)
    assert features["metadata_available"] is False
    for key, value in features.items():
        if key == "metadata_available":
            continue
        if isinstance(value, bool):
            assert value is False
        else:
            assert math.isnan(value)


def test_metadata_features_does_not_mutate_source():
    result = _make_metadata_result()
    before = copy.deepcopy(result.features)
    extract_metadata_features(result)
    assert result.features == before


# --- extract_forensic_features (combined) -------------------------------------


def test_combined_features_all_present():
    features = extract_forensic_features(
        ela_result=_make_ela_result(),
        copy_move_result=_make_copy_move_result(),
        metadata_result=_make_metadata_result(),
    )
    assert features["ela_available"] is True
    assert features["copy_move_available"] is True
    assert features["metadata_available"] is True
    assert set(features.keys()) == set(FORENSIC_FEATURE_NAMES)


def test_combined_features_all_missing():
    features = extract_forensic_features()
    assert features["ela_available"] is False
    assert features["copy_move_available"] is False
    assert features["metadata_available"] is False
    assert set(features.keys()) == set(FORENSIC_FEATURE_NAMES)


def test_combined_features_partial_availability():
    features = extract_forensic_features(ela_result=_make_ela_result())
    assert features["ela_available"] is True
    assert features["copy_move_available"] is False
    assert features["metadata_available"] is False
    assert set(features.keys()) == set(FORENSIC_FEATURE_NAMES)


def test_combined_feature_schema_is_stable_regardless_of_inputs():
    all_present = extract_forensic_features(
        _make_ela_result(), _make_copy_move_result(), _make_metadata_result()
    )
    all_missing = extract_forensic_features()
    assert set(all_present.keys()) == set(all_missing.keys())


def test_forensic_feature_names_no_duplicates():
    assert len(FORENSIC_FEATURE_NAMES) == len(set(FORENSIC_FEATURE_NAMES))


def test_forensic_feature_names_has_expected_prefixes():
    assert any(name.startswith("ela_") for name in FORENSIC_FEATURE_NAMES)
    assert any(name.startswith("copy_move_") for name in FORENSIC_FEATURE_NAMES)
    assert any(name.startswith("metadata_") for name in FORENSIC_FEATURE_NAMES)


def test_combined_features_is_deterministic():
    first = extract_forensic_features(
        _make_ela_result(), _make_copy_move_result(), _make_metadata_result()
    )
    second = extract_forensic_features(
        _make_ela_result(), _make_copy_move_result(), _make_metadata_result()
    )
    for key in first:
        v1, v2 = first[key], second[key]
        if isinstance(v1, float) and math.isnan(v1):
            assert isinstance(v2, float) and math.isnan(v2)
        else:
            assert v1 == v2


def test_no_fraud_labels_in_output():
    features = extract_forensic_features(
        _make_ela_result(), _make_copy_move_result(), _make_metadata_result()
    )
    forbidden_substrings = ("fraud", "label")
    for key in features:
        lowered = key.lower()
        assert not any(term in lowered for term in forbidden_substrings)
