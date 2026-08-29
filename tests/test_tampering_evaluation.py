"""Unit tests for src/forensics/tampering.py."""
import pytest
from src.forensics.copy_move import CopyMoveFeatures, CopyMoveResult
from src.forensics.ela import ELAFeatures, ELAResult
from src.forensics.metadata import MetadataFeatures, MetadataResult
from src.forensics.tampering import DocumentTamperingVerdict, evaluate_document_tampering


def _make_clean_metadata() -> MetadataResult:
    return MetadataResult(
        features=MetadataFeatures(
            width=800,
            height=600,
            aspect_ratio=1.33,
            file_size_bytes=102400,
            has_exif=True,
            exif_field_count=10,
            has_camera_info=True,
            has_software_tag=False,
            software_editing_indicator=False,
            has_datetime_original=True,
            has_datetime_modified=True,
            datetime_original_modified_mismatch=False,
            exif_malformed=False,
        ),
        structured_metadata={"format": "PDF", "page_count": 1},
        analysis={"indicators": []},
    )


def _make_clean_ela() -> ELAResult:
    return ELAResult(
        features=ELAFeatures(
            mean_error=2.0,
            std_error=1.0,
            median_error=1.5,
            max_error=10.0,
            p95_error=5.0,
            high_error_pixel_ratio=0.001,
        ),
        ela_image=None,
    )


def _make_clean_copy_move() -> CopyMoveResult:
    return CopyMoveResult(
        features=CopyMoveFeatures(
            num_keypoints=100,
            num_raw_matches=0,
            num_filtered_matches=0,
            match_ratio=0.0,
            mean_match_distance=0.0,
            std_match_distance=0.0,
            max_cluster_size=0,
        ),
        potential_duplicate_regions_found=False,
        visualization=None,
    )


def test_evaluate_tampering_clean_document():
    meta = _make_clean_metadata()
    ela = _make_clean_ela()
    cm = _make_clean_copy_move()

    verdict = evaluate_document_tampering(ela=ela, copy_move=cm, metadata=meta)
    assert isinstance(verdict, DocumentTamperingVerdict)
    assert verdict.status == "authentic"
    assert not verdict.is_tampered
    assert verdict.tampering_score < 0.20
    assert len(verdict.tampering_indicators) == 0


def test_evaluate_tampering_software_indicator():
    meta = MetadataResult(
        features=MetadataFeatures(
            width=800,
            height=600,
            aspect_ratio=1.33,
            file_size_bytes=102400,
            has_exif=True,
            exif_field_count=10,
            has_camera_info=False,
            has_software_tag=True,
            software_editing_indicator=True,
            has_datetime_original=True,
            has_datetime_modified=True,
            datetime_original_modified_mismatch=False,
            exif_malformed=False,
        ),
        structured_metadata={"format": "PDF", "software": "Canva"},
        analysis={"indicators": ["Canva detected"]},
    )
    verdict = evaluate_document_tampering(metadata=meta)
    assert verdict.status in ("suspicious", "tampered")
    assert verdict.is_tampered
    assert verdict.tampering_score >= 0.35
    assert any("Canva" in ind for ind in verdict.tampering_indicators)


def test_evaluate_tampering_date_mismatch():
    meta = MetadataResult(
        features=MetadataFeatures(
            width=800,
            height=600,
            aspect_ratio=1.33,
            file_size_bytes=102400,
            has_exif=True,
            exif_field_count=10,
            has_camera_info=False,
            has_software_tag=False,
            software_editing_indicator=False,
            has_datetime_original=True,
            has_datetime_modified=True,
            datetime_original_modified_mismatch=True,
            exif_malformed=False,
        ),
        structured_metadata={
            "format": "PDF",
            "datetime_original": "2026-01-01",
            "datetime_modified": "2026-02-01",
        },
        analysis={"indicators": []},
    )
    verdict = evaluate_document_tampering(metadata=meta)
    assert verdict.status == "suspicious"
    assert verdict.is_tampered
    assert any("mismatch" in ind.lower() for ind in verdict.tampering_indicators)


def test_evaluate_tampering_copy_move_cloning():
    cm = CopyMoveResult(
        features=CopyMoveFeatures(
            num_keypoints=200,
            num_raw_matches=40,
            num_filtered_matches=25,
            match_ratio=0.12,
            mean_match_distance=15.0,
            std_match_distance=5.0,
            max_cluster_size=6,
        ),
        potential_duplicate_regions_found=True,
        visualization=None,
    )
    verdict = evaluate_document_tampering(copy_move=cm)
    assert verdict.status in ("suspicious", "tampered")
    assert verdict.is_tampered
    assert any("duplicated" in ind.lower() or "cloned" in ind.lower() for ind in verdict.tampering_indicators)


def test_evaluate_tampering_high_ela_error():
    ela = ELAResult(
        features=ELAFeatures(
            mean_error=25.0,
            std_error=12.0,
            median_error=18.0,
            max_error=180.0,
            p95_error=65.0,
            high_error_pixel_ratio=0.08,
        ),
        ela_image=None,
    )
    verdict = evaluate_document_tampering(ela=ela)
    assert verdict.status == "suspicious"
    assert verdict.is_tampered
    assert any("compression" in ind.lower() or "variance" in ind.lower() for ind in verdict.tampering_indicators)


def test_evaluate_tampering_combined_severe_forgery():
    meta = MetadataResult(
        features=MetadataFeatures(
            width=800,
            height=600,
            aspect_ratio=1.33,
            file_size_bytes=102400,
            has_exif=True,
            exif_field_count=10,
            has_camera_info=False,
            has_software_tag=True,
            software_editing_indicator=True,
            has_datetime_original=True,
            has_datetime_modified=True,
            datetime_original_modified_mismatch=True,
            exif_malformed=False,
        ),
        structured_metadata={"format": "PDF", "software": "Adobe Photoshop CC 2024"},
        analysis={"indicators": []},
    )
    cm = CopyMoveResult(
        features=CopyMoveFeatures(
            num_keypoints=300,
            num_raw_matches=50,
            num_filtered_matches=30,
            match_ratio=0.10,
            mean_match_distance=20.0,
            std_match_distance=6.0,
            max_cluster_size=5,
        ),
        potential_duplicate_regions_found=True,
        visualization=None,
    )
    ela = ELAResult(
        features=ELAFeatures(
            mean_error=30.0,
            std_error=15.0,
            median_error=22.0,
            max_error=200.0,
            p95_error=75.0,
            high_error_pixel_ratio=0.10,
        ),
        ela_image=None,
    )

    verdict = evaluate_document_tampering(ela=ela, copy_move=cm, metadata=meta)
    assert verdict.status == "tampered"
    assert verdict.is_tampered
    assert verdict.tampering_score >= 0.70
    assert len(verdict.tampering_indicators) >= 3
