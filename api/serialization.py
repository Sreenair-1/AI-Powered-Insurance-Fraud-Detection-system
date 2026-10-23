"""Convert pandas/numpy/PIL values into JSON-safe Python values."""
from __future__ import annotations

import base64
import dataclasses
import io
import math
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# Longest edge of images returned to the browser. Overlays are drawn in the
# analyzed image's own pixel space, so downscaling the preview is display-only.
MAX_PREVIEW_EDGE = 1600


def to_jsonable(value: Any) -> Any:
    """Recursively turn numpy/pandas scalars and containers into plain JSON values.

    NaN/inf become `None` (strict JSON has no NaN). Unknown objects become `str`.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a `data:` URL, downscaled to `MAX_PREVIEW_EDGE`."""
    img = image
    if max(img.size) > MAX_PREVIEW_EDGE:
        img = img.copy()
        img.thumbnail((MAX_PREVIEW_EDGE, MAX_PREVIEW_EDGE))
    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=88) if fmt == "JPEG" else img.save(buf, format=fmt)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
