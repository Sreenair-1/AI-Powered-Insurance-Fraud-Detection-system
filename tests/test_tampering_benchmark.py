"""Unit tests for src/forensics/benchmark.py."""
import io
import json

import numpy as np
from PIL import Image

from src.forensics.benchmark import (
    BENIGN_TYPES,
    CONDITIONS,
    TAMPER_TYPES,
    detection_at_fpr,
    load_bill_folder,
    render_synthetic_bill,
    run_tampering_benchmark,
    samples_for_bill,
)
from src.forensics.metadata import analyze_metadata


def _bills(n=2, seed=0):
    rng = np.random.default_rng(seed)
    return [render_synthetic_bill(rng, name=f"b{i}") for i in range(n)], rng


def test_samples_cover_every_variant_and_condition_with_correct_labels():
    (bill, donor), rng = _bills()
    samples = samples_for_bill(bill, donor, rng)
    assert len(samples) == len(TAMPER_TYPES + BENIGN_TYPES) * len(CONDITIONS)
    for sample in samples:
        assert sample.is_tampered == (sample.variant in TAMPER_TYPES)
        assert sample.condition in CONDITIONS
        Image.open(io.BytesIO(sample.data)).load()  # every sample is a decodable image


def test_edits_change_pixels_and_metadata_as_documented():
    (bill, donor), rng = _bills()
    by_key = {(s.variant, s.condition): s.data for s in samples_for_bill(bill, donor, rng)}

    def pixels(key):
        return np.asarray(Image.open(io.BytesIO(by_key[key])).convert("RGB"), dtype=int)

    original = pixels(("original", "as_submitted"))
    for edit in TAMPER_TYPES:
        assert np.abs(pixels((edit, "as_submitted")) - original).max() > 100, edit

    camera = analyze_metadata(by_key[("original", "as_submitted")])
    assert camera.features.has_camera_info and not camera.features.software_editing_indicator
    editor = analyze_metadata(by_key[("amount_edit_in_editor", "as_submitted")])
    assert editor.features.software_editing_indicator and editor.features.datetime_original_modified_mismatch
    careful = analyze_metadata(by_key[("amount_edit", "as_submitted")])
    assert not careful.features.software_editing_indicator  # the careful forger keeps camera metadata
    reshared = analyze_metadata(by_key[("amount_edit_in_editor", "messaging_app")])
    assert not reshared.features.has_exif  # chat apps strip metadata


def test_detection_at_fpr_picks_best_threshold_within_budget():
    labels = [False, False, False, False, True, True, True, True]
    scores = [0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6, 0.95]
    best = detection_at_fpr(labels, scores, max_fpr=0.25)
    assert best == {"threshold": 0.4, "detection_rate": 1.0, "false_positive_rate": 0.25}
    assert detection_at_fpr(labels, scores, max_fpr=0.0)["detection_rate"] == 0.25
    assert detection_at_fpr([True, True], [0.1, 0.2])["threshold"] is None


def test_run_tampering_benchmark_report_is_json_serialisable():
    report = run_tampering_benchmark(n_bills=2, seed=1)
    json.dumps(report)
    assert report["n_samples"] == 2 * len(TAMPER_TYPES + BENIGN_TYPES) * len(CONDITIONS)
    for condition in CONDITIONS:
        block = report["results"][condition]
        assert block["n_tampered"] == block["n_untouched"] == 2 * len(TAMPER_TYPES)
        for method in ("ela", "copy_move", "metadata", "combined_verdict"):
            rates = block["methods"][method]
            assert 0.0 <= rates["detection_rate"] <= 1.0
            assert 0.0 <= rates["false_positive_rate"] <= 1.0
        assert set(block["detection_rate_by_edit"]) == set(TAMPER_TYPES)
    # The editor signature is detectable as submitted and gone after a chat re-share.
    by_edit = report["results"]["as_submitted"]["detection_rate_by_edit"]["amount_edit_in_editor"]
    assert by_edit["metadata"] == 1.0
    assert report["results"]["messaging_app"]["methods"]["metadata"]["detection_rate"] == 0.0


def test_folder_bills_are_used_when_supplied(tmp_path):
    rng = np.random.default_rng(2)
    for i in range(2):
        render_synthetic_bill(rng).image.save(tmp_path / f"real_{i}.jpg", quality=90)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    bills = load_bill_folder(tmp_path, np.random.default_rng(0))
    assert [b.name for b in bills] == ["real_0", "real_1"]
    assert all(b.source == "folder" and len(b.row_boxes) >= 2 for b in bills)
    report = run_tampering_benchmark(images_dir=tmp_path)
    assert report["source"] == "folder" and report["n_bills"] == 2
