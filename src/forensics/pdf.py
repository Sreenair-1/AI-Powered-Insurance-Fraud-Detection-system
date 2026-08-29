"""PDF Document Forensics and Error/Tampering Detection.

Extracts pages and metadata from claim-related PDF documents (such as invoices,
medical bills, discharge summaries, police reports, and repair estimates)
for forensic analysis including Error Level Analysis (ELA), Copy-Move Detection,
and PDF-level metadata and revision consistency checks.

Limitations (read before use):
- PDF page rendering converts vector and text content to raster bitmaps. ELA on
  a freshly rendered rasterized PDF page primarily highlights raster imagery
  and recompression artifacts within embedded elements or the rendered page.
- PDF metadata tags (Producer, Creator, timestamps) are plain text fields that
  can be modified, stripped, or spoofed. Their presence, absence, or mismatch
  is a forensic signal, not standalone proof of fraud.
- This module never decides or labels "fraud" from a threshold. Any such
  decision belongs in a downstream risk model.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional, Union

from PIL import Image
import pypdf
import pypdfium2 as pdfium

from src.forensics.metadata import (
    _EDITING_SOFTWARE_MARKERS as _BASE_EDITING_MARKERS,
    MetadataFeatures,
    MetadataResult,
)

PDFInput = Union[str, Path, bytes, bytearray, io.BytesIO, BinaryIO]

# Additional software markers commonly found in edited/modified PDF documents
_PDF_EDITING_SOFTWARE_MARKERS = _BASE_EDITING_MARKERS + (
    "acrobat",
    "pdf-xchange",
    "foxit",
    "nitro",
    "sejda",
    "ilovepdf",
    "smallpdf",
    "pdf24",
    "pdfescape",
    "soda pdf",
    "wondershare",
    "abbyy",
    "inkscape",
)


class PDFLoadError(ValueError):
    """Raised when an input PDF cannot be loaded, parsed, or is corrupted."""


@dataclass(frozen=True)
class PDFInfo:
    """Basic structural information about a PDF document."""

    page_count: int
    is_encrypted: bool
    file_size_bytes: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None


def _to_pdf_bytes(pdf_input: PDFInput) -> tuple[bytes, Optional[int]]:
    """Normalize various PDF inputs into raw bytes and optional file size."""
    if isinstance(pdf_input, (bytes, bytearray)):
        b = bytes(pdf_input)
        return b, len(b)

    if isinstance(pdf_input, (str, Path)):
        path = Path(pdf_input)
        if not path.exists():
            raise PDFLoadError(f"PDF file not found: {path}")
        try:
            b = path.read_bytes()
            return b, path.stat().st_size
        except OSError as exc:
            raise PDFLoadError(f"Could not read PDF file {path}: {exc}") from exc

    if hasattr(pdf_input, "read"):
        try:
            # Handle stream / UploadedFile
            if hasattr(pdf_input, "seek"):
                pdf_input.seek(0)
            b = pdf_input.read()
            if hasattr(pdf_input, "seek"):
                pdf_input.seek(0)
            size = len(b) if b else getattr(pdf_input, "size", None)
            return b, size
        except Exception as exc:
            raise PDFLoadError(f"Could not read PDF stream: {exc}") from exc

    raise PDFLoadError(f"Unsupported PDF input type: {type(pdf_input).__name__}")


def _open_pdfium(raw_bytes: bytes) -> pdfium.PdfDocument:
    """Open raw PDF bytes with pdfium, raising PDFLoadError on failure."""
    if not raw_bytes:
        raise PDFLoadError("PDF input is empty (0 bytes).")
    try:
        return pdfium.PdfDocument(raw_bytes)
    except Exception as exc:
        raise PDFLoadError(f"Failed to parse PDF with PDFium: {exc}") from exc


def _open_pypdf(raw_bytes: bytes) -> pypdf.PdfReader:
    """Open raw PDF bytes with pypdf, raising PDFLoadError on failure."""
    if not raw_bytes:
        raise PDFLoadError("PDF input is empty (0 bytes).")
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        return reader
    except Exception as exc:
        raise PDFLoadError(f"Failed to parse PDF with pypdf: {exc}") from exc


def get_pdf_info(pdf_input: PDFInput) -> PDFInfo:
    """Extract basic structural information and top-level tags from a PDF.

    Args:
        pdf_input: Path, bytes, or file-like stream representing a PDF.

    Returns:
        PDFInfo with page count, encryption status, size, and header tags.

    Raises:
        PDFLoadError: if the document is unreadable, empty, or corrupted.
    """
    raw_bytes, size = _to_pdf_bytes(pdf_input)
    reader = _open_pypdf(raw_bytes)

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise PDFLoadError(f"Could not read pages from PDF: {exc}") from exc

    meta = reader.metadata or {}
    return PDFInfo(
        page_count=page_count,
        is_encrypted=bool(reader.is_encrypted),
        file_size_bytes=size,
        title=str(meta.get("/Title", "")).strip() or None,
        author=str(meta.get("/Author", "")).strip() or None,
        creator=str(meta.get("/Creator", "")).strip() or None,
        producer=str(meta.get("/Producer", "")).strip() or None,
    )


def render_pdf_page(
    pdf_input: PDFInput,
    page_index: int = 0,
    scale: float = 2.0,
) -> Image.Image:
    """Render a single page of a PDF document as a high-resolution PIL RGB Image.

    Args:
        pdf_input: Path, bytes, or file-like stream representing a PDF.
        page_index: 0-indexed page number to render. Defaults to 0 (first page).
        scale: Multiplier for rendering resolution. Default 2.0 (~144 DPI)
            gives sharp text and crisp details for forensics.

    Returns:
        PIL.Image in RGB mode.

    Raises:
        PDFLoadError: if the document is unreadable, empty, or corrupted.
        ValueError: if page_index is out of range or scale is non-positive.
    """
    if not isinstance(page_index, int) or isinstance(page_index, bool):
        raise ValueError(f"page_index must be an int, got {type(page_index).__name__}")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError(f"scale must be a number, got {type(scale).__name__}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    raw_bytes, _ = _to_pdf_bytes(pdf_input)
    pdf = _open_pdfium(raw_bytes)

    page_count = len(pdf)
    if page_count == 0:
        raise PDFLoadError("PDF contains no pages.")
    if not 0 <= page_index < page_count:
        raise ValueError(
            f"page_index {page_index} out of range for PDF with {page_count} page(s) "
            f"(valid range: 0 to {page_count - 1})."
        )

    try:
        page = pdf[page_index]
        rendered = page.render(scale=scale).to_pil()
        # Ensure standard RGB format
        if rendered.mode != "RGB":
            rendered = rendered.convert("RGB")
        rendered.format = "PNG"
        return rendered
    except Exception as exc:
        raise PDFLoadError(f"Failed to render PDF page {page_index}: {exc}") from exc


def extract_pdf_text(
    pdf_input: PDFInput,
    page_index: Optional[int] = None,
) -> str:
    """Extract text content from a PDF document.

    Args:
        pdf_input: Path, bytes, or file-like stream representing a PDF.
        page_index: Optional 0-indexed page number to extract text from.
            If None, extracts text from all pages combined.

    Returns:
        Extracted text as a clean string.

    Raises:
        PDFLoadError: if the document is unreadable or corrupted.
        ValueError: if page_index is out of range.
    """
    raw_bytes, _ = _to_pdf_bytes(pdf_input)
    reader = _open_pypdf(raw_bytes)

    if page_index is not None:
        if not isinstance(page_index, int) or isinstance(page_index, bool):
            raise ValueError(f"page_index must be an int, got {type(page_index).__name__}")
        if not 0 <= page_index < len(reader.pages):
            raise ValueError(
                f"page_index {page_index} out of range for PDF with {len(reader.pages)} page(s)."
            )
        try:
            return reader.pages[page_index].extract_text() or ""
        except Exception:
            return ""

    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                texts.append(t)
        except Exception:
            pass
    return "\n\n".join(texts)


def _clean_pdf_date(raw_date: Optional[str]) -> Optional[str]:
    """Clean standard PDF date format (e.g. 'D:20260101120000+00\\'00\\'' -> '2026-01-01 12:00:00 UTC')."""
    if not raw_date or not isinstance(raw_date, str):
        return None
    s = raw_date.strip()
    if s.startswith("D:"):
        s = s[2:]
    # Remove apostrophes used in PDF timezone offsets
    s = s.replace("'", "")
    if len(s) >= 8 and s[:8].isdigit():
        year, month, day = s[:4], s[4:6], s[6:8]
        time_part = ""
        if len(s) >= 14 and s[8:14].isdigit():
            hour, minute, sec = s[8:10], s[10:12], s[12:14]
            time_part = f" {hour}:{minute}:{sec}"
        tz_part = s[14:] if len(s) > 14 else ""
        return f"{year}-{month}-{day}{time_part} {tz_part}".strip()
    return raw_date.strip() or None


def analyze_pdf_metadata(
    pdf_input: PDFInput,
    page_index: int = 0,
) -> MetadataResult:
    """Analyze structural and document metadata of a PDF for forensic indicators.

    Extracts PDF document catalog metadata (Author, Creator, Producer,
    CreationDate, ModDate) and maps them into standard `MetadataFeatures`
    and `MetadataResult` for seamless consumption by the feature fusion
    and explainable risk scoring pipeline.

    Args:
        pdf_input: Path, bytes, or file-like stream representing a PDF.
        page_index: Page index used to derive dimensions and aspect ratio.

    Returns:
        MetadataResult with structured metadata, numerical/boolean features,
        and descriptive forensic indicators.

    Raises:
        PDFLoadError: if the document is unreadable or corrupted.
    """
    raw_bytes, file_size_bytes = _to_pdf_bytes(pdf_input)
    reader = _open_pypdf(raw_bytes)

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise PDFLoadError(f"Could not read PDF pages: {exc}") from exc

    if page_count == 0:
        raise PDFLoadError("PDF contains no pages.")

    if not 0 <= page_index < page_count:
        raise ValueError(
            f"page_index {page_index} out of range for PDF with {page_count} page(s)."
        )

    # Extract page dimensions from media box
    try:
        page = reader.pages[page_index]
        width = int(float(page.mediabox.width))
        height = int(float(page.mediabox.height))
    except Exception:
        width, height = 0, 0

    aspect_ratio = (width / height) if height > 0 else 0.0

    meta = reader.metadata or {}
    raw_fields: dict[str, str] = {}
    malformed = False

    try:
        for k, v in meta.items():
            key_name = str(k).lstrip("/")
            raw_fields[key_name] = str(v)
    except Exception:
        malformed = True

    author = raw_fields.get("Author", "").strip()
    creator = raw_fields.get("Creator", "").strip()
    producer = raw_fields.get("Producer", "").strip()
    raw_creation_date = raw_fields.get("CreationDate", "").strip()
    raw_mod_date = raw_fields.get("ModDate", "").strip()
    title = raw_fields.get("Title", "").strip()
    subject = raw_fields.get("Subject", "").strip()

    software_tag = " / ".join(filter(None, [creator, producer])).strip()

    has_author_info = bool(author)
    has_software_tag = bool(software_tag)

    software_lower = software_tag.lower()
    software_editing_indicator = has_software_tag and any(
        marker in software_lower for marker in _PDF_EDITING_SOFTWARE_MARKERS
    )

    has_datetime_original = bool(raw_creation_date)
    has_datetime_modified = bool(raw_mod_date)
    datetime_mismatch = (
        has_datetime_original
        and has_datetime_modified
        and raw_creation_date != raw_mod_date
    )

    features = MetadataFeatures(
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        file_size_bytes=file_size_bytes,
        has_exif=len(raw_fields) > 0,
        exif_field_count=len(raw_fields),
        has_camera_info=has_author_info,
        has_software_tag=has_software_tag,
        software_editing_indicator=software_editing_indicator,
        has_datetime_original=has_datetime_original,
        has_datetime_modified=has_datetime_modified,
        datetime_original_modified_mismatch=datetime_mismatch,
        exif_malformed=malformed,
    )

    clean_creation_date = _clean_pdf_date(raw_creation_date)
    clean_mod_date = _clean_pdf_date(raw_mod_date)

    structured_metadata = {
        "format": "PDF",
        "page_count": page_count,
        "selected_page": page_index + 1,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "file_size_bytes": file_size_bytes,
        "is_encrypted": bool(reader.is_encrypted),
        "author": author or None,
        "title": title or None,
        "subject": subject or None,
        "creator": creator or None,
        "producer": producer or None,
        "software": software_tag or None,
        "datetime_original": clean_creation_date or raw_creation_date or None,
        "datetime_modified": clean_mod_date or raw_mod_date or None,
        "raw_metadata": raw_fields,
    }

    indicators: list[str] = []
    if not features.has_exif:
        indicators.append("No document metadata present in this PDF (metadata stripped or absent).")
    else:
        if software_editing_indicator:
            indicators.append(
                f"PDF Producer/Creator references editing software: '{software_tag}'."
            )
        if datetime_mismatch:
            indicators.append(
                f"PDF CreationDate ({clean_creation_date or raw_creation_date}) and "
                f"ModDate ({clean_mod_date or raw_mod_date}) differ, indicating post-creation edits."
            )
        if not has_author_info:
            indicators.append("No Author recorded in PDF metadata.")

    if reader.is_encrypted:
        indicators.append("PDF is encrypted / password-restricted.")

    if malformed:
        indicators.append("Some PDF metadata entries could not be parsed.")

    analysis = {
        "indicators": indicators,
        "note": (
            "PDF metadata is a forensic signal, not standalone proof of tampering. "
            "PDF editors, conversion tools, and print drivers legitimately alter "
            "metadata tags. This signal must be evaluated alongside other evidence."
        ),
    }

    return MetadataResult(
        structured_metadata=structured_metadata,
        features=features,
        analysis=analysis,
    )
