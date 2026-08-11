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
]
