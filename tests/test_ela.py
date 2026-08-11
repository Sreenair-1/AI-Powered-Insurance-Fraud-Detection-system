"""Unit tests for src/forensics/ela.py, using synthetic in-memory images only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.forensics.ela import ELAResult, ImageLoadError, compute_ela


def _make_image(mode: str = "RGB", size: tuple[int, int] = (64, 64)) -> Image.Image:
    rng = np.random.default_rng(42)
    if mode == "L":
        arr = rng.integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    elif mode == "RGBA":
        arr = rng.integers(0, 256, size=(size[1], size[0], 4), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


def _save(img: Image.Image, path: Path, fmt: str) -> Path:
    file_path = path / f"test.{fmt.lower()}"
    img.save(file_path, format=fmt)
    return file_path


def test_valid_jpeg_input(tmp_path: Path) -> None:
    path = _save(_make_image("RGB"), tmp_path, "JPEG")
    result = compute_ela(path)
    assert isinstance(result, ELAResult)


def test_valid_png_input(tmp_path: Path) -> None:
    path = _save(_make_image("RGB"), tmp_path, "PNG")
    result = compute_ela(path)
    assert isinstance(result, ELAResult)


def test_output_dimensions_match_input(tmp_path: Path) -> None:
    size = (80, 50)
    path = _save(_make_image("RGB", size), tmp_path, "PNG")
    result = compute_ela(path)
    assert result.ela_image.size == size


def test_output_is_valid_image(tmp_path: Path) -> None:
    path = _save(_make_image("RGB"), tmp_path, "PNG")
    result = compute_ela(path)
    assert result.ela_image.mode == "RGB"
    arr = np.asarray(result.ela_image)
    assert arr.dtype == np.uint8


@pytest.mark.parametrize("quality", [0, -1, 101, 1000])
def test_invalid_quality_rejected(tmp_path: Path, quality: int) -> None:
    path = _save(_make_image("RGB"), tmp_path, "PNG")
    with pytest.raises(ValueError):
        compute_ela(path, quality=quality)


@pytest.mark.parametrize("quality", [1, 50, 100])
def test_valid_quality_accepted(tmp_path: Path, quality: int) -> None:
    path = _save(_make_image("RGB"), tmp_path, "PNG")
    compute_ela(path, quality=quality)


@pytest.mark.parametrize("scale", [0, -1, -0.5])
def test_invalid_scale_rejected(tmp_path: Path, scale: float) -> None:
    path = _save(_make_image("RGB"), tmp_path, "PNG")
    with pytest.raises(ValueError):
        compute_ela(path, scale=scale)


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ImageLoadError):
        compute_ela(missing)


def test_corrupted_image_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "corrupted.jpg"
    bad_file.write_bytes(b"not a real image" * 10)
    with pytest.raises(ImageLoadError):
        compute_ela(bad_file)


def test_unsupported_file_type_raises(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("this is plain text, not an image")
    with pytest.raises(ImageLoadError):
        compute_ela(text_file)


def test_deterministic_features(tmp_path: Path) -> None:
    path = _save(_make_image("RGB"), tmp_path, "PNG")
    result1 = compute_ela(path, quality=85, scale=8)
    result2 = compute_ela(path, quality=85, scale=8)
    assert result1.features == result2.features


def test_original_file_not_modified(tmp_path: Path) -> None:
    path = _save(_make_image("RGB"), tmp_path, "PNG")
    original_bytes = path.read_bytes()
    compute_ela(path)
    assert path.read_bytes() == original_bytes


def test_features_are_finite(tmp_path: Path) -> None:
    path = _save(_make_image("RGB"), tmp_path, "PNG")
    result = compute_ela(path)
    for value in result.features.__dict__.values():
        assert np.isfinite(value)


def test_features_are_non_negative(tmp_path: Path) -> None:
    path = _save(_make_image("RGB"), tmp_path, "PNG")
    result = compute_ela(path)
    assert result.features.mean_error >= 0
    assert result.features.std_error >= 0
    assert result.features.median_error >= 0
    assert result.features.max_error >= 0
    assert result.features.p95_error >= 0
    assert 0 <= result.features.high_error_pixel_ratio <= 1


def test_grayscale_image_supported(tmp_path: Path) -> None:
    path = _save(_make_image("L"), tmp_path, "PNG")
    result = compute_ela(path)
    assert result.ela_image.mode == "RGB"


def test_rgba_image_supported(tmp_path: Path) -> None:
    path = _save(_make_image("RGBA"), tmp_path, "PNG")
    result = compute_ela(path)
    assert result.ela_image.mode == "RGB"


def test_very_small_image_supported(tmp_path: Path) -> None:
    path = _save(_make_image("RGB", (4, 4)), tmp_path, "PNG")
    result = compute_ela(path)
    assert result.ela_image.size == (4, 4)


def test_pil_image_input_accepted() -> None:
    img = _make_image("RGB")
    result = compute_ela(img)
    assert isinstance(result, ELAResult)


def test_metadata_contains_expected_keys(tmp_path: Path) -> None:
    path = _save(_make_image("RGB"), tmp_path, "JPEG")
    result = compute_ela(path, quality=75, scale=5)
    assert result.metadata["quality"] == 75
    assert result.metadata["scale"] == 5
    assert result.metadata["original_size"] == (64, 64)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_metadata_original_format_from_path(tmp_path: Path, fmt: str) -> None:
    path = _save(_make_image("RGB"), tmp_path, fmt)
    result = compute_ela(path)
    assert result.metadata["original_format"] == fmt
