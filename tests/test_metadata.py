"""Unit tests for src/forensics/metadata.py, using synthetic in-memory images only."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.forensics.metadata import ImageLoadError, MetadataResult, analyze_metadata


def _make_image(size: tuple[int, int] = (64, 48)) -> Image.Image:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save(img: Image.Image, path: Path, fmt: str, **save_kwargs) -> Path:
    file_path = path / f"test.{fmt.lower()}"
    img.save(file_path, format=fmt, **save_kwargs)
    return file_path


def _make_exif(fields: dict[int, str]) -> Image.Exif:
    exif = Image.Exif()
    for tag_id, value in fields.items():
        exif[tag_id] = value
    return exif


# EXIF tag ids used below.
_TAG_MAKE = 271
_TAG_MODEL = 272
_TAG_SOFTWARE = 305
_TAG_DATETIME = 306
_TAG_DATETIME_ORIGINAL = 36867


def test_valid_jpeg_input(tmp_path: Path) -> None:
    path = _save(_make_image(), tmp_path, "JPEG")
    result = analyze_metadata(path)
    assert isinstance(result, MetadataResult)
    assert result.features.width == 64
    assert result.features.height == 48


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "TIFF", "BMP"])
def test_supported_formats_load(tmp_path: Path, fmt: str) -> None:
    path = _save(_make_image(), tmp_path, fmt)
    result = analyze_metadata(path)
    assert result.structured_metadata["format"] == fmt


def test_image_with_no_exif_reports_absence(tmp_path: Path) -> None:
    path = _save(_make_image(), tmp_path, "PNG")
    result = analyze_metadata(path)
    assert result.features.has_exif is False
    assert result.features.exif_field_count == 0
    assert "No EXIF metadata present in this image." in result.analysis["indicators"]


def test_camera_and_software_exif_extracted(tmp_path: Path) -> None:
    exif = _make_exif(
        {
            _TAG_MAKE: "TestCam",
            _TAG_MODEL: "ModelX",
            _TAG_SOFTWARE: "Adobe Photoshop 24.0",
        }
    )
    path = _save(_make_image(), tmp_path, "JPEG", exif=exif)
    result = analyze_metadata(path)
    assert result.features.has_exif is True
    assert result.features.has_camera_info is True
    assert result.features.has_software_tag is True
    assert result.features.software_editing_indicator is True
    assert result.structured_metadata["camera_make"] == "TestCam"
    assert result.structured_metadata["camera_model"] == "ModelX"


def test_non_editing_software_not_flagged(tmp_path: Path) -> None:
    exif = _make_exif({_TAG_SOFTWARE: "CamFirmware 1.2"})
    path = _save(_make_image(), tmp_path, "JPEG", exif=exif)
    result = analyze_metadata(path)
    assert result.features.has_software_tag is True
    assert result.features.software_editing_indicator is False


def test_datetime_mismatch_flagged(tmp_path: Path) -> None:
    exif = _make_exif(
        {
            _TAG_DATETIME: "2020:01:01 00:00:00",
            _TAG_DATETIME_ORIGINAL: "2019:06:15 12:00:00",
        }
    )
    path = _save(_make_image(), tmp_path, "JPEG", exif=exif)
    result = analyze_metadata(path)
    assert result.features.has_datetime_modified is True
    assert result.features.has_datetime_original is True
    assert result.features.datetime_original_modified_mismatch is True


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ImageLoadError):
        analyze_metadata(missing)


def test_corrupted_file_raises(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not a real image file")
    with pytest.raises(ImageLoadError):
        analyze_metadata(corrupt)


def test_deterministic_output(tmp_path: Path) -> None:
    path = _save(_make_image(), tmp_path, "PNG")
    result1 = analyze_metadata(path)
    result2 = analyze_metadata(path)
    assert result1.structured_metadata == result2.structured_metadata
    assert result1.features == result2.features


def test_original_file_unchanged(tmp_path: Path) -> None:
    path = _save(_make_image(), tmp_path, "JPEG")
    before = path.read_bytes()
    analyze_metadata(path)
    after = path.read_bytes()
    assert before == after


def test_in_memory_image_has_no_file_size(tmp_path: Path) -> None:
    img = _make_image()
    result = analyze_metadata(img)
    assert result.features.file_size_bytes is None


def test_file_size_matches_actual_size(tmp_path: Path) -> None:
    path = _save(_make_image(), tmp_path, "PNG")
    result = analyze_metadata(path)
    assert result.features.file_size_bytes == path.stat().st_size


def test_feature_values_are_finite_and_valid(tmp_path: Path) -> None:
    path = _save(_make_image(), tmp_path, "JPEG")
    result = analyze_metadata(path)
    f = result.features
    assert f.width > 0
    assert f.height > 0
    assert math.isfinite(f.aspect_ratio)
    assert f.aspect_ratio > 0
    assert f.exif_field_count >= 0
    assert isinstance(f.has_exif, bool)
    assert isinstance(f.exif_malformed, bool)
