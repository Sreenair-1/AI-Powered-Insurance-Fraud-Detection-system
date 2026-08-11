"""Error Level Analysis (ELA) for claim-document images.

ELA re-saves an image at a controlled JPEG quality and measures the
pixel-level difference against the original. Regions that were edited or
pasted in after the image's last "true" compression tend to show a
different recompression error level than regions that were untouched,
because they have a different compression history.

This module produces an ELA visualization plus a small set of numerical
summary features intended for later consumption by a feature-fusion stage.

Limitations (read before use):
- ELA is a forensic *signal*, not standalone proof of image manipulation.
  Scans, screenshots, low-quality source images, and ordinary repeated
  JPEG saves can all produce elevated ELA error without any forgery.
- ELA here is JPEG-recompression-based. PNG/TIFF inputs are recompressed
  as JPEG purely to compute the signal; the result does not describe the
  source format's own (lossless) compression history.
- This module never decides or labels "fraud" from a threshold. Any such
  decision belongs in a downstream, explicitly-scoped risk model that
  combines this signal with others.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

ImageInput = Union[str, Path, Image.Image]

_SUPPORTED_FORMATS = {"JPEG", "PNG", "TIFF", "BMP"}

# Fixed absolute error threshold (0-255 scale, pre-`scale` amplification) used
# only to compute the descriptive `high_error_pixel_ratio` feature below. This
# is NOT a fraud/tamper decision threshold - it does not label anything as
# fraudulent, it just measures what fraction of a single image's own pixels
# have an unusually large recompression error relative to the rest of that
# image. Any fraud-relevant decision belongs in a downstream model.
_HIGH_ERROR_ABS_THRESHOLD = 30.0


class ImageLoadError(ValueError):
    """Raised when an input image cannot be loaded or is unsupported/corrupted."""


@dataclass(frozen=True)
class ELAFeatures:
    """Numerical summary of an ELA difference map. All values are non-negative."""

    mean_error: float
    std_error: float
    median_error: float
    max_error: float
    p95_error: float
    high_error_pixel_ratio: float


@dataclass(frozen=True)
class ELAResult:
    """Full output of an ELA run."""

    ela_image: Image.Image
    features: ELAFeatures
    metadata: dict = field(default_factory=dict)


def _load_image(image: ImageInput) -> Image.Image:
    """Load and validate an input image without mutating the source file."""
    if isinstance(image, Image.Image):
        img = image.copy()
    else:
        path = Path(image)
        if not path.exists():
            raise ImageLoadError(f"Image file not found: {path}")
        try:
            with Image.open(path) as opened:
                opened.load()
                img = opened.copy()
                img.format = opened.format
        except UnidentifiedImageError as exc:
            raise ImageLoadError(f"Could not identify image file: {path}") from exc
        except OSError as exc:
            raise ImageLoadError(f"Corrupted or unreadable image file: {path}") from exc

    fmt = (img.format or "").upper()
    if fmt and fmt not in _SUPPORTED_FORMATS:
        raise ImageLoadError(
            f"Unsupported image format: {fmt}. Supported formats: "
            f"{sorted(_SUPPORTED_FORMATS)}."
        )
    return img


def _validate_quality(quality: int) -> None:
    if not isinstance(quality, int) or isinstance(quality, bool):
        raise ValueError(f"quality must be an int, got {type(quality).__name__}")
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in [1, 100], got {quality}")


def _validate_scale(scale: float) -> None:
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError(f"scale must be a number, got {type(scale).__name__}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")


def _recompress_jpeg(rgb_image: Image.Image, quality: int) -> Image.Image:
    """Re-save an RGB image as JPEG at the given quality, entirely in memory."""
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        recompressed.load()
        return recompressed.copy()


def compute_ela(image: ImageInput, quality: int = 90, scale: float = 10.0) -> ELAResult:
    """Compute Error Level Analysis for a single claim-document image.

    Args:
        image: Path to an image file, or an already-loaded PIL Image.
            Supported source formats: JPEG, PNG, TIFF, BMP.
        quality: JPEG recompression quality used to derive the ELA signal,
            in [1, 100]. Default 90 (matches typical "already compressed"
            document scans, keeping the error signal in a normal range).
        scale: Multiplier applied to the raw pixel difference to make the
            ELA visualization perceptible. Does not affect the numerical
            features, which are computed from the unscaled difference.

    Returns:
        ELAResult with the ELA visualization image, numerical features, and
        run metadata (original format, size, quality, scale).

    Raises:
        ImageLoadError: input file missing, corrupted, or unsupported format.
        ValueError: quality or scale outside their valid ranges.

    Note:
        ELA output is a forensic indicator, not proof of tampering. See the
        module docstring for limitations.
    """
    _validate_quality(quality)
    _validate_scale(scale)

    original = _load_image(image)
    original_format = original.format
    original_size = original.size

    rgb_original = original.convert("RGB")
    rgb_recompressed = _recompress_jpeg(rgb_original, quality)

    original_arr = np.asarray(rgb_original, dtype=np.int16)
    recompressed_arr = np.asarray(rgb_recompressed, dtype=np.int16)
    diff = np.abs(original_arr - recompressed_arr).astype(np.float64)

    features = _extract_features(diff)

    scaled = np.clip(diff * scale, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(scaled, mode="RGB")

    metadata = {
        "original_format": original_format,
        "original_size": original_size,
        "quality": quality,
        "scale": scale,
    }
    return ELAResult(ela_image=ela_image, features=features, metadata=metadata)


def _extract_features(diff: np.ndarray) -> ELAFeatures:
    """Summarize a non-negative pixel-difference array into scalar features."""
    per_pixel_error = diff.mean(axis=-1)
    high_error_ratio = float(np.mean(per_pixel_error > _HIGH_ERROR_ABS_THRESHOLD))

    return ELAFeatures(
        mean_error=float(per_pixel_error.mean()),
        std_error=float(per_pixel_error.std()),
        median_error=float(np.median(per_pixel_error)),
        max_error=float(per_pixel_error.max()),
        p95_error=float(np.percentile(per_pixel_error, 95.0)),
        high_error_pixel_ratio=high_error_ratio,
    )
