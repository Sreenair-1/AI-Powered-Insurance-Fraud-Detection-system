"""Copy-Move Detection for claim-document images.

Copy-move forgery detection looks for regions *within the same image* that
duplicate another region of that same image (e.g. a form field pasted over
another field, or a stamp/signature duplicated to a second location). This
module detects candidate duplicated regions using ORB keypoint detection and
descriptor matching against the image itself, followed by filtering meant to
remove weak, ambiguous, or trivially-nearby matches.

Algorithm summary:
1. Detect ORB keypoints + binary descriptors on the grayscale image.
2. Match each descriptor against every other descriptor in the same image
   (self-matching) using a brute-force Hamming-distance matcher.
3. Discard the self-match (identical keypoint), then apply Lowe's ratio test
   between the best and second-best remaining candidates to discard
   ambiguous/weak matches.
4. Discard matches whose two keypoints are spatially closer than
   `min_distance` pixels (adjacent keypoints in a normal textured region
   commonly have similar descriptors; this is not evidence of copy-move).
5. Deduplicate symmetric pairs and summarize the remaining matches into
   numerical features, including the largest cluster of matches sharing an
   approximately equal spatial offset (a hallmark of a single block being
   copy-moved as a whole).

Limitations (read before use):
- This is a forensic *signal*, not standalone proof of image manipulation.
  Repetitive textures, patterned backgrounds, scanned form grids, and
  low-detail images can all produce matched keypoints without any forgery.
- ORB keypoint matching can miss copy-move forgeries that were rotated,
  heavily rescaled, or that occurred in low-texture/flat regions with too
  few keypoints to match.
- This module never decides or labels "fraud" from a threshold. Any such
  decision belongs in a downstream, explicitly-scoped risk model that
  combines this signal with others.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

ImageInput = Union[str, Path, Image.Image, np.ndarray, bytes, bytearray, io.BytesIO, BinaryIO]

_SUPPORTED_FORMATS = {"JPEG", "PNG", "TIFF", "BMP"}


class ImageLoadError(ValueError):
    """Raised when an input image cannot be loaded or is unsupported/corrupted."""


@dataclass(frozen=True)
class CopyMoveFeatures:
    """Numerical summary of a copy-move keypoint-matching run."""

    num_keypoints: int
    num_raw_matches: int
    num_filtered_matches: int
    match_ratio: float
    mean_match_distance: float
    std_match_distance: float
    max_cluster_size: int


@dataclass(frozen=True)
class CopyMoveResult:
    """Full output of a copy-move detection run."""

    potential_duplicate_regions_found: bool
    features: CopyMoveFeatures
    visualization: Optional[Image.Image]
    metadata: dict = field(default_factory=dict)


def _load_image(image: ImageInput) -> Image.Image:
    """Load and validate an input image without mutating the source."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            return Image.fromarray(arr.copy(), mode="L")
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), mode="RGB")
        if arr.ndim == 3 and arr.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA), mode="RGBA")
        raise ImageLoadError(f"Unsupported numpy array shape for image: {arr.shape}")

    if isinstance(image, Image.Image):
        img = image.copy()
        img.format = image.format
        return img

    raw_bytes: Optional[bytes] = None

    if isinstance(image, (bytes, bytearray)):
        raw_bytes = bytes(image)
    elif hasattr(image, "read"):
        try:
            if hasattr(image, "seek"):
                image.seek(0)
            raw_bytes = image.read()
            if hasattr(image, "seek"):
                image.seek(0)
        except Exception as exc:
            raise ImageLoadError(f"Could not read image stream: {exc}") from exc
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise ImageLoadError(f"Image file not found: {path}")
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()
        except OSError as exc:
            raise ImageLoadError(f"Corrupted or unreadable image file: {path}") from exc
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
    return img


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_positive_float(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_ratio_threshold(value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"ratio_threshold must be a number, got {type(value).__name__}")
    if not 0 < value <= 1:
        raise ValueError(f"ratio_threshold must be in (0, 1], got {value}")


def _find_matches(
    keypoints,
    descriptors: Optional[np.ndarray],
    ratio_threshold: float,
    min_distance: float,
) -> tuple[int, list[tuple[int, int, float]]]:
    """Self-match descriptors and filter to strong, spatially-separated pairs.

    Returns (raw_match_count, filtered_matches). `filtered_matches` entries
    are `(keypoint_index_lo, keypoint_index_hi, spatial_distance_px)`.
    """
    if descriptors is None or len(descriptors) < 2:
        return 0, []

    k = min(3, len(descriptors))
    if k < 2:
        return 0, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(descriptors, descriptors, k=k)

    raw_count = 0
    seen_pairs: set[tuple[int, int]] = set()
    filtered: list[tuple[int, int, float]] = []

    for group in knn:
        if not group:
            continue
        query_idx = group[0].queryIdx
        candidates = sorted(
            (m for m in group if m.trainIdx != query_idx),
            key=lambda m: m.distance,
        )
        if len(candidates) < 2:
            continue
        raw_count += 1

        best, second = candidates[0], candidates[1]
        if second.distance <= 0:
            continue
        if best.distance >= ratio_threshold * second.distance:
            continue

        i, j = query_idx, best.trainIdx
        lo, hi = (i, j) if i < j else (j, i)
        if (lo, hi) in seen_pairs:
            continue

        pt_lo = keypoints[lo].pt
        pt_hi = keypoints[hi].pt
        spatial_dist = float(np.hypot(pt_hi[0] - pt_lo[0], pt_hi[1] - pt_lo[1]))
        if spatial_dist < min_distance:
            continue

        seen_pairs.add((lo, hi))
        filtered.append((lo, hi, spatial_dist))

    return raw_count, filtered


def _max_offset_cluster(
    matches: list[tuple[int, int, float]], keypoints, bin_size: float
) -> int:
    """Size of the largest group of matches sharing an approximately equal
    spatial offset — many matches with the same offset indicate a whole
    block was translated, not just isolated similar-looking keypoints."""
    counts: dict[tuple[int, int], int] = {}
    for lo, hi, _ in matches:
        pt_lo = keypoints[lo].pt
        pt_hi = keypoints[hi].pt
        dx = pt_hi[0] - pt_lo[0]
        dy = pt_hi[1] - pt_lo[1]
        key = (round(dx / bin_size), round(dy / bin_size))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def _extract_features(
    matches: list[tuple[int, int, float]],
    num_keypoints: int,
    raw_count: int,
    keypoints,
    offset_bin_size: float,
) -> CopyMoveFeatures:
    num_filtered = len(matches)
    if num_filtered == 0:
        return CopyMoveFeatures(
            num_keypoints=num_keypoints,
            num_raw_matches=raw_count,
            num_filtered_matches=0,
            match_ratio=0.0,
            mean_match_distance=0.0,
            std_match_distance=0.0,
            max_cluster_size=0,
        )

    distances = np.array([d for _, _, d in matches], dtype=np.float64)
    return CopyMoveFeatures(
        num_keypoints=num_keypoints,
        num_raw_matches=raw_count,
        num_filtered_matches=num_filtered,
        match_ratio=float(num_filtered / num_keypoints) if num_keypoints else 0.0,
        mean_match_distance=float(distances.mean()),
        std_match_distance=float(distances.std()),
        max_cluster_size=_max_offset_cluster(matches, keypoints, offset_bin_size),
    )


def _build_visualization(
    rgb_image: Image.Image, keypoints, matches: list[tuple[int, int, float]]
) -> Image.Image:
    """Draw lines between matched keypoint pairs on a copy of the image."""
    vis_rgb = np.array(rgb_image, dtype=np.uint8).copy()
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    for lo, hi, _ in matches:
        pt1 = tuple(int(round(c)) for c in keypoints[lo].pt)
        pt2 = tuple(int(round(c)) for c in keypoints[hi].pt)
        cv2.line(vis_bgr, pt1, pt2, color=(0, 0, 255), thickness=1)
        cv2.circle(vis_bgr, pt1, 3, (0, 255, 0), -1)
        cv2.circle(vis_bgr, pt2, 3, (0, 255, 0), -1)
    return Image.fromarray(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), mode="RGB")


def compute_copy_move(
    image: ImageInput,
    max_keypoints: int = 500,
    ratio_threshold: float = 0.75,
    min_distance: float = 20.0,
    offset_bin_size: float = 5.0,
    visualize: bool = True,
) -> CopyMoveResult:
    """Detect candidate copy-moved (duplicated) regions within a single image.

    Args:
        image: Path to an image file, an already-loaded PIL Image, or an
            OpenCV-style numpy array (grayscale, BGR, or BGRA). Supported
            source file formats: JPEG, PNG, TIFF, BMP.
        max_keypoints: Maximum ORB keypoints to detect (`ORB_create(nfeatures=...)`).
        ratio_threshold: Lowe's-ratio-test threshold in (0, 1] used to reject
            ambiguous matches; lower is stricter. Default 0.75.
        min_distance: Minimum pixel distance between two keypoints for their
            match to be kept; rejects trivially-nearby matches from ordinary
            local texture. Must be > 0.
        offset_bin_size: Bin width (pixels) used to cluster matches by their
            spatial offset when computing `max_cluster_size`.
        visualize: If True, also render a visualization image with matched
            keypoint pairs connected by lines.

    Returns:
        CopyMoveResult with a boolean indicating whether any candidate
        duplicated regions were found, numerical features, an optional
        visualization image, and run metadata.

    Raises:
        ImageLoadError: input missing, corrupted, or unsupported format/type.
        ValueError: a parameter is outside its valid range.

    Note:
        This is a forensic indicator, not proof of tampering. See the module
        docstring for limitations.
    """
    _validate_positive_int("max_keypoints", max_keypoints)
    _validate_ratio_threshold(ratio_threshold)
    _validate_positive_float("min_distance", min_distance)
    _validate_positive_float("offset_bin_size", offset_bin_size)

    original = _load_image(image)
    original_format = original.format
    original_size = original.size
    original_mode = original.mode

    gray = np.asarray(original.convert("L"), dtype=np.uint8)

    orb = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    num_keypoints = len(keypoints) if keypoints is not None else 0

    if num_keypoints >= 2 and descriptors is not None:
        raw_count, matches = _find_matches(
            keypoints, descriptors, ratio_threshold, min_distance
        )
    else:
        raw_count, matches = 0, []

    features = _extract_features(
        matches, num_keypoints, raw_count, keypoints, offset_bin_size
    )

    visualization = None
    if visualize:
        visualization = _build_visualization(
            original.convert("RGB"), keypoints if keypoints is not None else [], matches
        )

    metadata = {
        "original_format": original_format,
        "original_size": original_size,
        "original_mode": original_mode,
        "max_keypoints": max_keypoints,
        "ratio_threshold": ratio_threshold,
        "min_distance": min_distance,
        "offset_bin_size": offset_bin_size,
    }

    return CopyMoveResult(
        potential_duplicate_regions_found=len(matches) > 0,
        features=features,
        visualization=visualization,
        metadata=metadata,
    )
