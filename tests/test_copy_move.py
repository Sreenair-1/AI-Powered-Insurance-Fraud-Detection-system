"""Unit tests for src/forensics/copy_move.py, using synthetic in-memory images only."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.forensics.copy_move import CopyMoveResult, ImageLoadError, compute_copy_move


def _noise_image(size: tuple[int, int] = (200, 200), seed: int = 7) -> np.ndarray:
    """RGB uint8 array with enough texture for ORB to find keypoints."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)


def _copy_moved_image(size: tuple[int, int] = (200, 200), seed: int = 7) -> Image.Image:
    """Noise image with a textured patch duplicated to a distant location."""
    arr = _noise_image(size, seed)
    patch = arr[20:60, 20:60].copy()
    arr[120:160, 120:160] = patch
    return Image.fromarray(arr, mode="RGB")


def _plain_image(size: tuple[int, int] = (16, 16), color: int = 128) -> Image.Image:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save(img: Image.Image, path: Path, fmt: str) -> Path:
    file_path = path / f"test.{fmt.lower()}"
    img.save(file_path, format=fmt)
    return file_path


def test_valid_image_returns_result(tmp_path: Path) -> None:
    path = _save(_copy_moved_image(), tmp_path, "PNG")
    result = compute_copy_move(path)
    assert isinstance(result, CopyMoveResult)


def test_detects_duplicated_patch() -> None:
    img = _copy_moved_image()
    result = compute_copy_move(img)
    assert result.features.num_filtered_matches > 0
    assert result.potential_duplicate_regions_found is True


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ImageLoadError):
        compute_copy_move(missing)


def test_corrupted_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a real image")
    with pytest.raises(ImageLoadError):
        compute_copy_move(bad)


def test_invalid_input_type_raises() -> None:
    with pytest.raises(ImageLoadError):
        compute_copy_move(12345)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_keypoints": 0},
        {"max_keypoints": -5},
        {"ratio_threshold": 0},
        {"ratio_threshold": 1.5},
        {"min_distance": 0},
        {"min_distance": -1},
        {"offset_bin_size": 0},
    ],
)
def test_invalid_parameters_rejected(kwargs: dict) -> None:
    img = _copy_moved_image()
    with pytest.raises(ValueError):
        compute_copy_move(img, **kwargs)


def test_deterministic_behavior() -> None:
    img = _copy_moved_image()
    result_a = compute_copy_move(img)
    result_b = compute_copy_move(img)
    assert result_a.features == result_b.features
    assert result_a.potential_duplicate_regions_found == result_b.potential_duplicate_regions_found
    arr_a = np.asarray(result_a.visualization)
    arr_b = np.asarray(result_b.visualization)
    assert np.array_equal(arr_a, arr_b)


def test_simple_uniform_image_does_not_crash() -> None:
    img = _plain_image()
    result = compute_copy_move(img)
    assert isinstance(result, CopyMoveResult)
    assert result.features.num_filtered_matches == 0
    assert result.potential_duplicate_regions_found is False


def test_tiny_image_does_not_crash() -> None:
    img = _plain_image(size=(4, 4))
    result = compute_copy_move(img)
    assert isinstance(result, CopyMoveResult)


def test_numerical_features_are_finite() -> None:
    img = _copy_moved_image()
    result = compute_copy_move(img)
    for value in (
        result.features.num_keypoints,
        result.features.num_raw_matches,
        result.features.num_filtered_matches,
        result.features.match_ratio,
        result.features.mean_match_distance,
        result.features.std_match_distance,
        result.features.max_cluster_size,
    ):
        assert math.isfinite(value)


def test_visualization_dimensions_and_type() -> None:
    size = (200, 180)
    img = _copy_moved_image(size)
    result = compute_copy_move(img)
    assert isinstance(result.visualization, Image.Image)
    assert result.visualization.mode == "RGB"
    assert result.visualization.size == size


def test_visualization_disabled_returns_none() -> None:
    img = _copy_moved_image()
    result = compute_copy_move(img, visualize=False)
    assert result.visualization is None


def test_original_pil_image_unchanged() -> None:
    img = _copy_moved_image()
    before = np.array(img).copy()
    compute_copy_move(img)
    after = np.array(img)
    assert np.array_equal(before, after)


def test_original_ndarray_unchanged() -> None:
    arr = _noise_image()
    before = arr.copy()
    compute_copy_move(arr)
    assert np.array_equal(before, arr)


def test_ndarray_input_accepted() -> None:
    arr = _noise_image()
    result = compute_copy_move(arr)
    assert isinstance(result, CopyMoveResult)


def test_grayscale_ndarray_input_accepted() -> None:
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, size=(100, 100), dtype=np.uint8)
    result = compute_copy_move(arr)
    assert isinstance(result, CopyMoveResult)


def test_matched_pairs_metadata_matches_filtered_count() -> None:
    result = compute_copy_move(_copy_moved_image())
    pairs = result.metadata["matched_pairs"]
    assert len(pairs) == result.features.num_filtered_matches
    assert all(len(p) == 4 for p in pairs)
