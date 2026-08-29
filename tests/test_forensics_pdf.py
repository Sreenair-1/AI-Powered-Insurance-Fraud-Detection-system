"""Unit tests for PDF document forensics (src/forensics/pdf.py).

Uses synthetic in-memory PDFs created dynamically via pypdf (no committed binary
files) to verify page extraction, rendering, metadata parsing, editing software
marker detection, timestamp consistency checks, and error handling.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pypdf
import pytest
from PIL import Image

from src.features.forensic import FORENSIC_FEATURE_NAMES, extract_forensic_features
from src.forensics.copy_move import compute_copy_move
from src.forensics.ela import compute_ela
from src.forensics.pdf import (
    PDFInfo,
    PDFLoadError,
    analyze_pdf_metadata,
    get_pdf_info,
    render_pdf_page,
)


def _create_synthetic_pdf(
    num_pages: int = 1,
    metadata: dict[str, str] | None = None,
    page_size: tuple[float, float] = (300.0, 400.0),
) -> bytes:
    """Create a minimal synthetic PDF in memory with optional metadata."""
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=page_size[0], height=page_size[1])

    if metadata:
        formatted_meta = {
            (k if k.startswith("/") else f"/{k}"): v for k, v in metadata.items()
        }
        writer.add_metadata(formatted_meta)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# get_pdf_info tests
# ---------------------------------------------------------------------------


def test_get_pdf_info_valid_in_memory():
    pdf_bytes = _create_synthetic_pdf(
        num_pages=3,
        metadata={
            "Title": "Claim Receipt",
            "Author": "Hospital A",
            "Creator": "Billing Software v1",
            "Producer": "PDF Generator",
        },
    )

    info = get_pdf_info(pdf_bytes)

    assert isinstance(info, PDFInfo)
    assert info.page_count == 3
    assert not info.is_encrypted
    assert info.title == "Claim Receipt"
    assert info.author == "Hospital A"
    assert info.creator == "Billing Software v1"
    assert info.producer == "PDF Generator"
    assert info.file_size_bytes == len(pdf_bytes)


def test_get_pdf_info_from_file_path(tmp_path: Path):
    pdf_bytes = _create_synthetic_pdf(num_pages=2)
    path = tmp_path / "test_doc.pdf"
    path.write_bytes(pdf_bytes)

    info = get_pdf_info(path)
    assert info.page_count == 2
    assert info.file_size_bytes == len(pdf_bytes)


def test_get_pdf_info_from_stream():
    pdf_bytes = _create_synthetic_pdf(num_pages=1)
    stream = io.BytesIO(pdf_bytes)

    info = get_pdf_info(stream)
    assert info.page_count == 1


def test_get_pdf_info_missing_file_raises(tmp_path: Path):
    with pytest.raises(PDFLoadError, match="PDF file not found"):
        get_pdf_info(tmp_path / "missing.pdf")


def test_get_pdf_info_empty_input_raises():
    with pytest.raises(PDFLoadError, match="empty"):
        get_pdf_info(b"")


def test_get_pdf_info_corrupted_input_raises():
    with pytest.raises(PDFLoadError):
        get_pdf_info(b"not a real pdf content")


# ---------------------------------------------------------------------------
# render_pdf_page tests
# ---------------------------------------------------------------------------


def test_render_pdf_page_valid():
    pdf_bytes = _create_synthetic_pdf(num_pages=2, page_size=(200, 300))

    img_p0 = render_pdf_page(pdf_bytes, page_index=0, scale=1.0)
    assert isinstance(img_p0, Image.Image)
    assert img_p0.mode == "RGB"
    assert img_p0.size == (200, 300)

    # Scale 2.0
    img_p1 = render_pdf_page(pdf_bytes, page_index=1, scale=2.0)
    assert img_p1.size == (400, 600)


def test_render_pdf_page_out_of_range_raises():
    pdf_bytes = _create_synthetic_pdf(num_pages=2)
    with pytest.raises(ValueError, match="out of range"):
        render_pdf_page(pdf_bytes, page_index=5)

    with pytest.raises(ValueError, match="out of range"):
        render_pdf_page(pdf_bytes, page_index=-1)


def test_render_pdf_page_invalid_scale_raises():
    pdf_bytes = _create_synthetic_pdf(num_pages=1)
    with pytest.raises(ValueError, match="scale must be positive"):
        render_pdf_page(pdf_bytes, scale=0.0)

    with pytest.raises(ValueError, match="scale must be a number"):
        render_pdf_page(pdf_bytes, scale="invalid")  # type: ignore


# ---------------------------------------------------------------------------
# analyze_pdf_metadata tests
# ---------------------------------------------------------------------------


def test_analyze_pdf_metadata_clean_document():
    pdf_bytes = _create_synthetic_pdf(
        num_pages=2,
        metadata={
            "Author": "Authorized Hospital",
            "Creator": "Hospital Management System",
            "Producer": "Official PDF Writer",
            "CreationDate": "D:20260101100000+00'00'",
            "ModDate": "D:20260101100000+00'00'",
        },
        page_size=(612.0, 792.0),
    )

    result = analyze_pdf_metadata(pdf_bytes, page_index=0)

    features = result.features
    assert features.width == 612
    assert features.height == 792
    assert pytest.approx(features.aspect_ratio, rel=1e-3) == 612 / 792
    assert features.has_exif
    assert features.exif_field_count >= 4
    assert features.has_camera_info  # author present
    assert features.has_software_tag
    assert not features.software_editing_indicator
    assert features.has_datetime_original
    assert features.has_datetime_modified
    assert not features.datetime_original_modified_mismatch
    assert not features.exif_malformed

    assert result.structured_metadata["format"] == "PDF"
    assert result.structured_metadata["page_count"] == 2
    assert result.structured_metadata["selected_page"] == 1


def test_analyze_pdf_metadata_editing_software_detected():
    pdf_bytes = _create_synthetic_pdf(
        num_pages=1,
        metadata={
            "Creator": "Adobe Photoshop 2024 (Windows)",
            "Producer": "Canva Editor",
            "CreationDate": "D:20260101100000+00'00'",
            "ModDate": "D:20260105153000+00'00'",
        },
    )

    result = analyze_pdf_metadata(pdf_bytes)

    features = result.features
    assert features.has_software_tag
    assert features.software_editing_indicator
    assert features.datetime_original_modified_mismatch

    indicators = result.analysis.get("indicators", [])
    assert any("editing software" in ind for ind in indicators)
    assert any("differ" in ind for ind in indicators)


def test_analyze_pdf_metadata_empty_metadata():
    raw_pdf_no_info = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >> endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer << /Size 4 /Root 1 0 R >>\n"
        b"startxref\n190\n%%EOF\n"
    )

    result = analyze_pdf_metadata(raw_pdf_no_info)

    features = result.features
    assert not features.has_exif
    assert features.exif_field_count == 0
    assert not features.has_camera_info
    assert not features.has_software_tag
    assert not features.software_editing_indicator
    assert not features.has_datetime_original
    assert not features.has_datetime_modified


def test_analyze_pdf_metadata_out_of_range_page_raises():
    pdf_bytes = _create_synthetic_pdf(num_pages=2)
    with pytest.raises(ValueError, match="out of range"):
        analyze_pdf_metadata(pdf_bytes, page_index=10)


# ---------------------------------------------------------------------------
# End-to-end Forensic checks with PDF
# ---------------------------------------------------------------------------


def test_pdf_forensics_feature_fusion_compatibility():
    pdf_bytes = _create_synthetic_pdf(
        num_pages=1,
        metadata={
            "Creator": "Nitro Pro 13",
            "CreationDate": "D:20260101100000+00'00'",
            "ModDate": "D:20260102100000+00'00'",
        },
    )

    page_img = render_pdf_page(pdf_bytes, page_index=0, scale=1.0)
    ela_res = compute_ela(page_img)
    cm_res = compute_copy_move(page_img)
    meta_res = analyze_pdf_metadata(pdf_bytes)

    assert ela_res is not None
    assert cm_res is not None
    assert meta_res is not None

    fused_features = extract_forensic_features(
        ela_result=ela_res,
        copy_move_result=cm_res,
        metadata_result=meta_res,
    )

    # Output schema stability check
    assert tuple(fused_features.keys()) == FORENSIC_FEATURE_NAMES
    assert fused_features["ela_available"] is True
    assert fused_features["copy_move_available"] is True
    assert fused_features["metadata_available"] is True
    assert fused_features["metadata_software_editing_indicator"] is True
    assert fused_features["metadata_datetime_original_modified_mismatch"] is True
