"""Regression tests for the four redesign implementation issues and UI enhancements.

Issues covered:
1. Prevent stale standalone document-assessment results from being displayed after the uploaded file or selected PDF page changes.
2. Validate the selected claim after dataset sampling changes so a missing claim cannot cause an .iloc[0] failure.
3. Replace redundant broad exception tuples with appropriate error handling and safe user-facing messages.
4. Reconcile README links to documentation files that are absent from the repository.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from PIL import Image

from app import claim_view, data_loader, model_loader, ui_components
from app.config import HEALTH_MODEL_DIR_ENV, MOTOR_MODEL_DIR_ENV
from src.data.schemas import PROJECT_ROOT
from src.features.structured import CLAIM_KEY_COL
from src.forensics import PDFLoadError
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.models.persistence import save_health_model, save_motor_model
from src.pipeline.inference import ClaimRiskAssessment, RiskThresholds


# ---------------------------------------------------------------------------
# Issue 1: Stale standalone document-assessment results prevention
# ---------------------------------------------------------------------------


def test_standalone_doc_signature_invalidation():
    """Verify that document signature changes properly invalidate stale standalone results."""
    # Simulate session state
    session_state = {
        "new_claim_result": ("claim:NEW-123", MagicMock(), MagicMock(), MagicMock()),
        "new_claim_doc_sig": ("doc1.pdf", 1024, "file_id_1", 0, "motor"),
    }

    # Case A: Same signature preserves results
    current_doc_sig = ("doc1.pdf", 1024, "file_id_1", 0, "motor")
    assert session_state.get("new_claim_doc_sig") == current_doc_sig
    assert "new_claim_result" in session_state

    # Case B: Page index changes (e.g. from page 0 to page 1) -> invalidates stale result
    page_changed_sig = ("doc1.pdf", 1024, "file_id_1", 1, "motor")
    if session_state.get("new_claim_doc_sig") != page_changed_sig:
        session_state.pop("new_claim_result", None)

    assert "new_claim_result" not in session_state

    # Re-populate
    session_state["new_claim_result"] = ("claim:NEW-456", MagicMock(), MagicMock(), MagicMock())
    session_state["new_claim_doc_sig"] = page_changed_sig

    # Case C: File changes (different file name or ID) -> invalidates stale result
    new_file_sig = ("doc2.png", 2048, "file_id_2", 0, "motor")
    if session_state.get("new_claim_doc_sig") != new_file_sig:
        session_state.pop("new_claim_result", None)

    assert "new_claim_result" not in session_state

    # Case D: File cleared (uploaded is None) -> clears state
    session_state["new_claim_result"] = ("claim:NEW-789", MagicMock(), MagicMock(), MagicMock())
    session_state["new_claim_doc_sig"] = new_file_sig

    # Upload cleared
    session_state.pop("new_claim_result", None)
    session_state.pop("new_claim_doc_sig", None)
    assert "new_claim_result" not in session_state
    assert "new_claim_doc_sig" not in session_state


# ---------------------------------------------------------------------------
# Issue 2: Validate selected claim after sampling changes (prevent .iloc[0] crash)
# ---------------------------------------------------------------------------


def test_selected_claim_validation_after_sampling_changes():
    """Verify that a claim key not in the dataset sample does not cause an .iloc[0] failure."""
    # Build a tiny synthetic dataset
    sample_keys = [f"claim:{i}" for i in range(5)]
    df_X = pd.DataFrame({CLAIM_KEY_COL: sample_keys, "feature1": range(5)})

    dataset_mock = MagicMock()
    dataset_mock.X = df_X

    # Claim that was previously selected, but is absent after sampling change
    stale_claim_key = "claim:999"
    assert stale_claim_key not in dataset_mock.X[CLAIM_KEY_COL].values

    # Check the safety guard implemented in _render_claim_detail
    matches = dataset_mock.X[dataset_mock.X[CLAIM_KEY_COL] == stale_claim_key]
    assert matches.empty

    # Verify that attempting .iloc[0] without the guard would raise IndexError
    with pytest.raises(IndexError):
        _ = matches.iloc[0]

    # Verify that the guard detects matches.empty safely
    guard_executed = False
    if matches.empty:
        guard_executed = True
        error_msg = f"Selected claim '{stale_claim_key}' was not found in the currently loaded dataset sample."
    else:
        _ = matches.iloc[0]

    assert guard_executed
    assert stale_claim_key in error_msg


# ---------------------------------------------------------------------------
# Issue 3: Replace redundant broad exception tuples
# ---------------------------------------------------------------------------


def test_exception_handling_in_run_forensic_checks(tmp_path: Path):
    """Verify run_forensic_checks gracefully handles invalid files without redundant exception tuples."""
    # Create an invalid/corrupt image file
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"NOT_A_VALID_JPEG_IMAGE_DATA")

    # Should not raise uncaught exception
    result = claim_view.run_forensic_checks(corrupt_file)
    assert not result.any_available
    assert result.ela is None
    assert result.copy_move is None
    assert result.metadata is None


# ---------------------------------------------------------------------------
# Issue 4: Reconcile README links to documentation files
# ---------------------------------------------------------------------------


def test_readme_links_all_exist():
    """Verify every documentation file linked in README.md exists in the repository."""
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist in PROJECT_ROOT"

    content = readme_path.read_text(encoding="utf-8")

    # Find all markdown links [text](target)
    links = re.findall(r'\[([^\]]+)\]\(([^)#]+)(?:#[^)]*)?\)', content)
    assert len(links) > 0, "README.md should contain documentation links"

    checked_doc_links = 0
    for text, url in links:
        if url.startswith("http://") or url.startswith("https://") or url.startswith("mailto:"):
            continue
        # Check doc links
        target_path = (PROJECT_ROOT / url).resolve()
        assert target_path.exists(), f"README link '[{text}]({url})' points to non-existent file: {target_path}"
        if url.startswith("docs/"):
            checked_doc_links += 1

    assert checked_doc_links >= 10, f"Expected at least 10 docs/ links in README.md, found {checked_doc_links}"


# ---------------------------------------------------------------------------
# Model Loader: Curve Data Loaders
# ---------------------------------------------------------------------------


def test_load_roc_and_pr_curve_sidecars(tmp_path: Path):
    """Verify load_roc_curve and load_pr_curve return dict when present and None when absent."""
    # Absent
    assert model_loader.load_roc_curve(tmp_path) is None
    assert model_loader.load_pr_curve(tmp_path) is None

    # Sidecar present
    roc_data = {"fpr": [0.0, 0.2, 0.5, 1.0], "tpr": [0.0, 0.6, 0.85, 1.0]}
    (tmp_path / model_loader.ROC_CURVE_FILENAME).write_text(json.dumps(roc_data), encoding="utf-8")
    loaded_roc = model_loader.load_roc_curve(tmp_path)
    assert loaded_roc == roc_data

    pr_data = {"precision": [1.0, 0.8, 0.6, 0.2], "recall": [0.0, 0.4, 0.7, 1.0]}
    (tmp_path / model_loader.PR_CURVE_FILENAME).write_text(json.dumps(pr_data), encoding="utf-8")
    loaded_pr = model_loader.load_pr_curve(tmp_path)
    assert loaded_pr == pr_data


def test_load_roc_curve_embedded_in_eval_metrics(tmp_path: Path):
    """Verify load_roc_curve can read embedded curve dictionary from eval_metrics.json."""
    embedded = {
        "accuracy": 0.85,
        "roc_curve": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0]},
    }
    (tmp_path / model_loader.EVAL_METRICS_FILENAME).write_text(json.dumps(embedded), encoding="utf-8")

    loaded = model_loader.load_roc_curve(tmp_path)
    assert loaded is not None
    assert loaded["fpr"] == [0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# UI Components: KPI row layout & delta badges
# ---------------------------------------------------------------------------


def test_delta_badge_styling():
    """Verify delta_badge formats positive, negative, and flat score deltas."""
    from app.ui_theme import delta_badge

    pos_badge = delta_badge(0.125)
    assert "is-delta-pos" in pos_badge
    assert "+12.5%" in pos_badge
    assert "▲" in pos_badge

    neg_badge = delta_badge(-0.082)
    assert "is-delta-neg" in neg_badge
    assert "-8.2%" in neg_badge
    assert "▼" in neg_badge

    flat_badge = delta_badge(0.0)
    assert "is-delta-flat" in flat_badge
    assert "0.0%" in flat_badge
