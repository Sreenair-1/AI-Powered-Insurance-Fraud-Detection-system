"""Metadata Analysis for claim-document images.

Reads whatever structural and EXIF metadata is actually embedded in an image
(format, dimensions, color mode, camera/software tags, timestamps) and turns
it into a small set of presence/absence and consistency indicators.

Limitations (read before use):
- This is a forensic *signal*, not standalone proof of image manipulation.
  Missing EXIF is normal for screenshots, scans, and many legitimately
  re-saved or messaging-app-compressed images - it is not evidence of
  tampering by itself.
- EXIF fields are simple, easily editable text/number tags. Their presence,
  absence, or internal consistency is only weak circumstantial evidence.
- This module never decides or labels "fraud" from a threshold. Any such
  decision belongs in a downstream, explicitly-scoped risk model that
  combines this signal with others.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional, Union

from PIL import ExifTags, Image, UnidentifiedImageError

ImageInput = Union[str, Path, Image.Image, bytes, bytearray, io.BytesIO, BinaryIO]

_SUPPORTED_FORMATS = {"JPEG", "PNG", "TIFF", "BMP"}

# Substrings (checked case-insensitively) that indicate the EXIF Software tag
# names a general-purpose image editor rather than camera/scanner firmware.
# This is a weak, easily-spoofed indicator - never a fraud decision by itself.
_EDITING_SOFTWARE_MARKERS = (
    "photoshop",
    "gimp",
    "paint.net",
    "paintshop",
    "illustrator",
    "lightroom",
    "snapseed",
    "pixlr",
    "affinity photo",
    "corel",
    "picsart",
    "canva",
)


class ImageLoadError(ValueError):
    """Raised when an input image cannot be loaded or is unsupported/corrupted."""


@dataclass(frozen=True)
class MetadataFeatures:
    """Numerical/boolean forensic features derived from image metadata."""

    width: int
    height: int
    aspect_ratio: float
    file_size_bytes: Optional[int]
    has_exif: bool
    exif_field_count: int
    has_camera_info: bool
    has_software_tag: bool
    software_editing_indicator: bool
    has_datetime_original: bool
    has_datetime_modified: bool
    datetime_original_modified_mismatch: bool
    exif_malformed: bool


@dataclass(frozen=True)
class MetadataResult:
    """Full output of a metadata analysis run."""

    structured_metadata: dict
    features: MetadataFeatures
    analysis: dict = field(default_factory=dict)


def _load_image(image: ImageInput) -> tuple[Image.Image, Optional[int]]:
    """Load and validate an input image without mutating the source file.

    Returns the loaded image plus the source file size in bytes (``None`` for
    an already-in-memory ``PIL.Image`` with no known file size).
    """
    if isinstance(image, Image.Image):
        img = image.copy()
        img.format = image.format
        return img, None

    raw_bytes: Optional[bytes] = None
    file_size: Optional[int] = None

    if isinstance(image, (bytes, bytearray)):
        raw_bytes = bytes(image)
        file_size = len(raw_bytes)
    elif hasattr(image, "read"):
        try:
            if hasattr(image, "seek"):
                image.seek(0)
            raw_bytes = image.read()
            if hasattr(image, "seek"):
                image.seek(0)
            file_size = len(raw_bytes)
        except Exception as exc:
            raise ImageLoadError(f"Could not read image stream: {exc}") from exc
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise ImageLoadError(f"Image file not found: {path}")
        try:
            file_size = path.stat().st_size
            with open(path, "rb") as f:
                raw_bytes = f.read()
        except OSError as exc:
            raise ImageLoadError(f"Could not read image file: {path}") from exc
    else:
        raise ImageLoadError(f"Unsupported image input type: {type(image).__name__}")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as opened:
            opened.load()
            img = opened.copy()
            img.format = opened.format
    except UnidentifiedImageError as exc:
        raise ImageLoadError("Could not identify image file format.") from exc
    except Exception as exc:
        raise ImageLoadError(f"Corrupted or unreadable image: {exc}") from exc

    fmt = (img.format or "").upper()
    if fmt and fmt not in _SUPPORTED_FORMATS:
        raise ImageLoadError(
            f"Unsupported image format: {fmt}. Supported formats: "
            f"{sorted(_SUPPORTED_FORMATS)}."
        )
    return img, file_size


def _stringify_exif_value(value: object) -> str:
    """Render an EXIF tag value as a plain, JSON/log-safe string."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip("\x00").strip()
        except Exception:
            return value.hex()
    return str(value)


def _extract_exif(img: Image.Image) -> tuple[dict[str, str], bool]:
    """Extract EXIF tags (top-level IFD0 + Exif sub-IFD) as a flat name->str dict.

    Returns ``(fields, malformed)`` where ``malformed`` is set if any part of
    the EXIF block raised while being parsed (corrupt/truncated data). Parsing
    proceeds on a best-effort basis - a malformed sub-block does not prevent
    other, well-formed fields from being returned.
    """
    fields: dict[str, str] = {}
    malformed = False

    try:
        exif = img.getexif()
    except Exception:
        return fields, True

    if not exif:
        return fields, False

    try:
        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            fields[tag_name] = _stringify_exif_value(value)
    except Exception:
        malformed = True

    try:
        sub_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        for tag_id, value in sub_ifd.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            fields.setdefault(tag_name, _stringify_exif_value(value))
    except Exception:
        malformed = True

    return fields, malformed


def analyze_metadata(image: ImageInput) -> MetadataResult:
    """Analyze the structural and EXIF metadata of a single claim-document image.

    Args:
        image: Path to an image file, or an already-loaded PIL Image.
            Supported source formats: JPEG, PNG, TIFF, BMP.

    Returns:
        MetadataResult with structured metadata, numerical/boolean forensic
        features, and a short descriptive (not accusatory) analysis.

    Raises:
        ImageLoadError: input file missing, corrupted, or unsupported format.

    Note:
        Metadata is a forensic indicator, not proof of tampering. See the
        module docstring for limitations.
    """
    img, file_size_bytes = _load_image(image)

    width, height = img.size
    exif_fields, exif_malformed = _extract_exif(img)

    camera_make = exif_fields.get("Make", "").strip()
    camera_model = exif_fields.get("Model", "").strip()
    software = exif_fields.get("Software", "").strip()
    datetime_original = exif_fields.get("DateTimeOriginal", "").strip()
    datetime_modified = exif_fields.get("DateTime", "").strip()

    has_camera_info = bool(camera_make or camera_model)
    has_software_tag = bool(software)
    software_editing_indicator = bool(software) and any(
        marker in software.lower() for marker in _EDITING_SOFTWARE_MARKERS
    )
    has_datetime_original = bool(datetime_original)
    has_datetime_modified = bool(datetime_modified)
    datetime_mismatch = (
        has_datetime_original
        and has_datetime_modified
        and datetime_original != datetime_modified
    )

    features = MetadataFeatures(
        width=width,
        height=height,
        aspect_ratio=(width / height) if height else 0.0,
        file_size_bytes=file_size_bytes,
        has_exif=len(exif_fields) > 0,
        exif_field_count=len(exif_fields),
        has_camera_info=has_camera_info,
        has_software_tag=has_software_tag,
        software_editing_indicator=software_editing_indicator,
        has_datetime_original=has_datetime_original,
        has_datetime_modified=has_datetime_modified,
        datetime_original_modified_mismatch=datetime_mismatch,
        exif_malformed=exif_malformed,
    )

    structured_metadata = {
        "format": img.format,
        "width": width,
        "height": height,
        "color_mode": img.mode,
        "file_size_bytes": file_size_bytes,
        "camera_make": camera_make or None,
        "camera_model": camera_model or None,
        "software": software or None,
        "datetime_original": datetime_original or None,
        "datetime_modified": datetime_modified or None,
        "exif_fields": exif_fields,
    }

    indicators: list[str] = []
    if not features.has_exif:
        indicators.append("No EXIF metadata present in this image.")
    else:
        if not has_camera_info:
            indicators.append("EXIF present but no camera Make/Model recorded.")
        if software_editing_indicator:
            indicators.append(
                f"Software tag references known image-editing software: '{software}'."
            )
        if datetime_mismatch:
            indicators.append(
                "EXIF DateTimeOriginal and DateTime fields differ."
            )
    if exif_malformed:
        indicators.append("Some EXIF data could not be parsed (malformed/corrupt fields).")

    analysis = {
        "indicators": indicators,
        "note": (
            "Metadata presence, absence, or inconsistency is a weak, easily "
            "spoofed forensic signal. It must be combined with other evidence "
            "and must never be used alone to declare an image fraudulent."
        ),
    }

    return MetadataResult(
        structured_metadata=structured_metadata, features=features, analysis=analysis
    )
