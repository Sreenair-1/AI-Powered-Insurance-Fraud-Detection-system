"""Document/image forensics components (ELA, copy-move, metadata)."""
from src.forensics.copy_move import (
    CopyMoveFeatures,
    CopyMoveResult,
)
from src.forensics.copy_move import ImageLoadError as CopyMoveImageLoadError
from src.forensics.copy_move import compute_copy_move
from src.forensics.ela import ELAFeatures, ELAResult, ImageLoadError, compute_ela
from src.forensics.metadata import ImageLoadError as MetadataImageLoadError
from src.forensics.metadata import MetadataFeatures, MetadataResult, analyze_metadata
from src.forensics.pdf import (
    PDFInfo,
    PDFLoadError,
    analyze_pdf_metadata,
    extract_pdf_text,
    get_pdf_info,
    render_pdf_page,
)
from src.forensics.tampering import (
    DocumentTamperingVerdict,
    evaluate_document_tampering,
)

__all__ = [
    "compute_ela",
    "ELAResult",
    "ELAFeatures",
    "ImageLoadError",
    "compute_copy_move",
    "CopyMoveResult",
    "CopyMoveFeatures",
    "CopyMoveImageLoadError",
    "analyze_metadata",
    "MetadataResult",
    "MetadataFeatures",
    "MetadataImageLoadError",
    "render_pdf_page",
    "extract_pdf_text",
    "analyze_pdf_metadata",
    "get_pdf_info",
    "PDFInfo",
    "PDFLoadError",
    "DocumentTamperingVerdict",
    "evaluate_document_tampering",
]
