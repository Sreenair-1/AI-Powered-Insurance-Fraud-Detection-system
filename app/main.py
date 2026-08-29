"""Streamlit decision-support UI for AI-powered insurance claim fraud detection.

Thin renderer over `app.model_loader` (persisted models), `app.data_loader`
(sampled, fused claim data), and `app.claim_view` (assessment + optional
forensic re-scoring) -- see `docs/app_ui.md` for the full architecture and
`docs/pipeline.md` for what `src/pipeline/inference.py` actually computes.

"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.claim_view import (
    ClaimViewError,
    assess_claim,
    assess_new_claim_document,
    claim_row,
    generate_new_claim_id,
    rescore_with_forensics,
    run_forensic_checks,
)
from app.data_loader import (
    DatasetNotAvailableError,
    get_cached_health_dataset,
    get_cached_motor_dataset,
    health_claim_row,
    motor_claim_row,
)
from app.model_loader import ModelNotAvailableError, get_health_model, get_motor_model, load_eval_metrics
from src.features.structured import CLAIM_KEY_COL
from src.forensics import (
    DocumentTamperingVerdict,
    PDFLoadError,
    evaluate_document_tampering,
    extract_pdf_text,
    get_pdf_info,
    render_pdf_page,
)
from src.pipeline.inference import DEFAULT_RISK_THRESHOLDS

st.set_page_config(page_title="Insurance Claim Fraud Risk Review", layout="wide")

MOTOR_DOMAIN = "motor"
HEALTH_DOMAIN = "health"

_RISK_ICON = {"low": "\U0001F7E2", "medium": "\U0001F7E1", "high": "\U0001F534"}


def _render_header() -> None:
    st.title("Insurance Claim Fraud Risk Review")
    st.caption(
        "Decision-support only. Risk scores and SHAP explanations describe model "
        "behavior, not proof of fraud -- a \"high\" result means **flagged for "
        "investigation**, never a determination"
    )


def _sidebar_domain_and_sampling() -> tuple[str, dict]:
    st.sidebar.header("Dataset")
    domain = st.sidebar.radio("Claim type", (MOTOR_DOMAIN, HEALTH_DOMAIN), format_func=str.title)
    st.sidebar.caption(
        "Claims are loaded from a bounded, seeded sample of the real dataset: "
        "exact graph centrality (`src/graph/features.py`) does not scale to the "
        "full dataset within an interactive session. See docs/app_ui.md."
    )
    params: dict = {}
    if domain == MOTOR_DOMAIN:
        params["sample_size"] = st.sidebar.slider(
            "Motor claims to load",
            100,
            config.MAX_MOTOR_SAMPLE_SIZE,
            config.DEFAULT_MOTOR_SAMPLE_SIZE,
            step=100,
        )
    else:
        params["inpatient_sample_size"] = st.sidebar.slider(
            "Inpatient claims to load",
            20,
            config.MAX_HEALTH_INPATIENT_SAMPLE_SIZE,
            config.DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE,
            step=20,
        )
        params["outpatient_sample_size"] = st.sidebar.slider(
            "Outpatient claims to load",
            40,
            config.MAX_HEALTH_OUTPATIENT_SAMPLE_SIZE,
            config.DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE,
            step=40,
        )
    return domain, params


def _load_model(domain: str):
    directory = config.motor_model_dir() if domain == MOTOR_DOMAIN else config.health_model_dir()
    getter = get_motor_model if domain == MOTOR_DOMAIN else get_health_model
    try:
        return getter(str(directory)), directory
    except ModelNotAvailableError as exc:
        st.error(str(exc))
        st.stop()
        raise  # unreachable, keeps type-checkers happy


def _load_dataset(domain: str, params: dict):
    try:
        if domain == MOTOR_DOMAIN:
            return get_cached_motor_dataset(**params)
        return get_cached_health_dataset(**params)
    except DatasetNotAvailableError as exc:
        st.error(
            f"{exc}\n\nThis app needs the raw datasets under `data/raw/` -- see "
            "docs/data_contract.md for the expected local file layout."
        )
        st.stop()
        raise  # unreachable, keeps type-checkers happy


def _render_claim_info(domain: str, dataset, claim_key: str) -> None:
    row = motor_claim_row(dataset, claim_key) if domain == MOTOR_DOMAIN else health_claim_row(dataset, claim_key)
    st.subheader("Claim information")
    st.dataframe(row.to_frame(name="value"), use_container_width=True)


def _render_graph_panel(dataset, claim_key: str) -> None:
    row = dataset.X[dataset.X[CLAIM_KEY_COL] == claim_key].iloc[0]
    st.subheader("Graph relationship signals")
    cols = st.columns(4)
    cols[0].metric("Shared entities", "Yes" if bool(row.get("has_shared_entity")) else "No")
    cols[1].metric("Max entity claim degree", int(row.get("max_entity_claim_degree", 0)))
    cols[2].metric("Community ID", int(row.get("community_id", -1)))
    cols[3].metric("Degree centrality", f"{float(row.get('degree_centrality', 0.0)):.4f}")
    st.caption(
        "A high 'shared entities'/centrality value means this claim's "
        "provider/garage/bank-account etc. is also linked to other claims -- "
        "evidence to weigh, not proof of a fraud ring."
    )


def _render_assessment(assessment) -> None:
    summary = assessment.to_ui_summary()
    icon = _RISK_ICON.get(summary["risk_category"], "")
    cols = st.columns(2)
    cols[0].metric("Fraud probability", f"{summary['fraud_probability']:.1%}")
    cols[1].markdown(f"### {icon} {summary['risk_category_description']}")

    st.markdown("**Top contributing factors (SHAP)**")
    factors_df = pd.DataFrame(summary["top_factors"])
    if not factors_df.empty:
        factors_df = factors_df.rename(
            columns={"feature": "Feature", "feature_value": "Claim value", "shap_value": "SHAP contribution"}
        ).set_index("Feature")
        st.dataframe(factors_df, use_container_width=True)
        st.bar_chart(factors_df["SHAP contribution"])
    st.caption(
        "Positive SHAP contributions push the prediction toward higher fraud "
        "risk; negative contributions push it lower. This explains the "
        "model's behavior, not a finding of fraud."
    )


def _render_forensics_panel(domain: str, model, dataset, claim_key: str) -> None:
    st.subheader("Document forensics (optional)")
    uploaded = st.file_uploader(
        "Upload a claim document (image or PDF) to check for tampering/errors (ELA / copy-move / metadata)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"],
    )
    if uploaded is None:
        st.caption("No document uploaded for this claim -- forensic features are treated as unavailable.")
        return

    is_pdf = uploaded.name.lower().endswith(".pdf") or (
        hasattr(uploaded, "type") and uploaded.type == "application/pdf"
    )

    page_index = 0
    preview_image = None
    extracted_text = ""

    file_size_kb = len(uploaded.getvalue()) / 1024.0 if hasattr(uploaded, "getvalue") else 0.0

    if is_pdf:
        try:
            pdf_info = get_pdf_info(uploaded)
            if pdf_info.page_count > 1:
                page_selection = st.selectbox(
                    "Select PDF page to analyze",
                    options=list(range(1, pdf_info.page_count + 1)),
                    format_func=lambda p: f"Page {p} of {pdf_info.page_count}",
                )
                page_index = page_selection - 1
            preview_image = render_pdf_page(uploaded, page_index=page_index, scale=1.5)
            extracted_text = extract_pdf_text(uploaded, page_index=page_index)
            forensic = run_forensic_checks(uploaded, page_index=page_index)
        except (PDFLoadError, Exception) as exc:
            st.warning(f"Could not read or render the uploaded PDF file: {exc}")
            return
    else:
        try:
            preview_image = Image.open(uploaded)
            forensic = run_forensic_checks(preview_image)
        except Exception:
            st.warning("Could not read the uploaded file as an image.")
            return

    if not forensic.any_available:
        st.warning("Could not analyze the uploaded file; forensic features remain unavailable.")
        return

    # Document Overview Metrics
    st.markdown("#### 📄 Uploaded Document Summary")
    meta_dict = forensic.metadata.structured_metadata if forensic.metadata else {}
    doc_cols = st.columns(4)
    doc_cols[0].metric("File Name", uploaded.name)
    doc_cols[1].metric("File Size", f"{file_size_kb:.1f} KB")
    doc_cols[2].metric(
        "Format",
        f"PDF ({meta_dict.get('page_count', 1)} pages)" if is_pdf else str(meta_dict.get("format", "Image")),
    )
    doc_cols[3].metric(
        "Software Detected",
        "Yes (Editing Tool)" if forensic.metadata and forensic.metadata.features.software_editing_indicator else ("Present" if meta_dict.get("software") else "None"),
    )

    # Document Preview & Extracted Content Expander
    with st.expander("Document Visual Preview & Extracted Text", expanded=False):
        if preview_image is not None:
            st.image(
                preview_image,
                caption=f"Preview: {uploaded.name} (Page {page_index + 1})" if is_pdf else f"Preview: {uploaded.name}",
                use_container_width=True,
            )
        if extracted_text.strip():
            st.markdown("**Extracted Document Text Content:**")
            st.text_area("Text extracted from document", extracted_text, height=150, disabled=True)

    # Three-Column Forensic Analysis
    st.markdown("#### 🔬 Document Tampering & Forensics Inspection")
    col1, col2, col3 = st.columns(3)
    if forensic.ela is not None:
        col1.markdown("**1. Error Level Analysis (ELA)**")
        col1.image(forensic.ela.ela_image, use_container_width=True)
        col1.metric("High-error pixel ratio", f"{forensic.ela.features.high_error_pixel_ratio:.3f}")
        col1.caption("Measures compression variance across pixel regions. High localized variance highlights spliced or pasted content.")
    else:
        col1.info("ELA unavailable for this file.")

    if forensic.copy_move is not None:
        col2.markdown("**2. Copy-Move Forgery Detection**")
        if forensic.copy_move.visualization is not None:
            col2.image(forensic.copy_move.visualization, use_container_width=True)
        col2.metric(
            "Duplicated Regions",
            "Detected" if forensic.copy_move.potential_duplicate_regions_found else "None Found",
        )
        col2.caption("Uses ORB keypoint matching to detect duplicated signatures, hospital stamps, or cloned amounts.")
    else:
        col2.info("Copy-move detection unavailable for this file.")

    if forensic.metadata is not None:
        col3.markdown("**3. Metadata & Revision Analysis**")
        if is_pdf:
            if meta_dict.get("author"):
                col3.write(f"• **Author:** {meta_dict.get('author')}")
            if meta_dict.get("software"):
                col3.write(f"• **Software Tag:** `{meta_dict.get('software')}`")
            if meta_dict.get("datetime_original"):
                col3.write(f"• **Created:** {meta_dict.get('datetime_original')}")
            if meta_dict.get("datetime_modified"):
                col3.write(f"• **Modified:** {meta_dict.get('datetime_modified')}")
        else:
            col3.write(f"• **Dimensions:** {meta_dict.get('width', 0)} × {meta_dict.get('height', 0)} px")
            if meta_dict.get("camera_make") or meta_dict.get("camera_model"):
                cam = " ".join(filter(None, [meta_dict.get("camera_make"), meta_dict.get("camera_model")]))
                col3.write(f"• **Camera:** {cam}")
            if meta_dict.get("software"):
                col3.write(f"• **Software Tag:** `{meta_dict.get('software')}`")
            if meta_dict.get("datetime_original"):
                col3.write(f"• **Created:** {meta_dict.get('datetime_original')}")
            if meta_dict.get("datetime_modified"):
                col3.write(f"• **Modified:** {meta_dict.get('datetime_modified')}")

        st_indicators = forensic.metadata.analysis.get("indicators", [])
        if st_indicators:
            col3.markdown("**Forensic Indicators:**")
            for indicator in st_indicators:
                col3.write(f"- {indicator}")
        col3.caption(forensic.metadata.analysis.get("note", ""))
    else:
        col3.info("Metadata analysis unavailable for this file.")

    # Re-scoring integration
    try:
        rescored = rescore_with_forensics(domain, model, dataset, claim_key, forensic)
    except ClaimViewError as exc:
        st.error(str(exc))
        return

    st.markdown("---")
    st.subheader("🎯 Claim Risk Assessment Updated with Document Evidence")
    st.caption("How this uploaded document's forensic evidence fused with the structured claim and graph signals:")
    _render_assessment(rescored)


def _render_eval_metrics(model_dir) -> None:
    with st.expander("Model evaluation (held-out validation split, from training time)"):
        metrics = load_eval_metrics(model_dir)
        if metrics is None:
            st.info(
                "No evaluation metrics recorded for this artifact. Re-run "
                "`scripts/train_models.py` to generate one."
            )
            return
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        cols[1].metric("Precision", f"{metrics['precision']:.3f}")
        cols[2].metric("Recall", f"{metrics['recall']:.3f}")
        cols[3].metric("F1", f"{metrics['f1']:.3f}")
        cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        st.caption(f"{metrics['n_samples']} validation claims ({metrics['n_positive']} positive).")


def _render_new_claim_tab(domain: str, model) -> None:
    st.subheader("📄 Upload & Analyze New Claim Document")
    st.caption(
        "Upload a new claim document (invoice, medical bill, discharge summary, repair estimate, police report). "
        "The system will check the document for digital tampering, assign a unique Claim ID, and compute the fraud risk score."
    )

    uploaded = st.file_uploader(
        "Upload a claim document to inspect (PDF, JPEG, PNG, TIFF, BMP)",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        key="new_claim_uploader",
    )

    if uploaded is None:
        st.info("👆 Upload a document above to generate a new claim and check for tampering.")
        return

    is_pdf = uploaded.name.lower().endswith(".pdf") or (
        hasattr(uploaded, "type") and uploaded.type == "application/pdf"
    )

    page_index = 0
    preview_image = None
    extracted_text = ""
    file_size_kb = len(uploaded.getvalue()) / 1024.0 if hasattr(uploaded, "getvalue") else 0.0

    if is_pdf:
        try:
            pdf_info = get_pdf_info(uploaded)
            if pdf_info.page_count > 1:
                page_selection = st.selectbox(
                    "Select PDF page to analyze",
                    options=list(range(1, pdf_info.page_count + 1)),
                    format_func=lambda p: f"Page {p} of {pdf_info.page_count}",
                    key="new_claim_pdf_page",
                )
                page_index = page_selection - 1
            preview_image = render_pdf_page(uploaded, page_index=page_index, scale=1.5)
            extracted_text = extract_pdf_text(uploaded, page_index=page_index)
        except (PDFLoadError, Exception) as exc:
            st.warning(f"Could not read or render the uploaded PDF file: {exc}")
            return
    else:
        try:
            preview_image = Image.open(uploaded)
        except Exception:
            st.warning("Could not read the uploaded file as an image.")
            return

    # Run standalone assessment
    try:
        new_claim_id, forensic, verdict, assessment = assess_new_claim_document(
            domain=domain,
            model=model,
            document=uploaded,
            page_index=page_index,
        )
    except Exception as exc:
        st.error(f"Error during claim assessment: {exc}")
        return

    # Generated Claim ID Card
    st.success(f"🎫 Generated New Claim ID: **`{new_claim_id}`** (Domain: **{domain.title()} Insurance**)")

    # Tampering Verdict Card
    st.markdown("### 🛡️ Document Tampering & Integrity Verdict")
    if verdict.status == "tampered":
        st.error(
            f"🚨 **DOCUMENT TAMPERING STATUS: HIGH RISK / TAMPERED** "
            f"(Tampering Anomaly Score: {verdict.tampering_score:.0%})"
        )
    elif verdict.status == "suspicious":
        st.warning(
            f"⚠️ **DOCUMENT TAMPERING STATUS: SUSPICIOUS / ELEVATED ANOMALIES** "
            f"(Tampering Anomaly Score: {verdict.tampering_score:.0%})"
        )
    else:
        st.success(
            f"✅ **DOCUMENT TAMPERING STATUS: AUTHENTIC / LOW RISK** "
            f"(Tampering Anomaly Score: {verdict.tampering_score:.0%})"
        )

    st.write(verdict.summary)

    if verdict.tampering_indicators:
        st.markdown("**Tampering Red Flags Detected:**")
        for flag in verdict.tampering_indicators:
            st.markdown(f"- 🔴 **{flag}**")
    else:
        st.markdown("- 🟢 No digital alteration markers, software signatures, or duplicated regions detected.")

    # Document Overview Metrics
    st.markdown("#### 📄 Document Information & Extracted Content")
    meta_dict = forensic.metadata.structured_metadata if forensic.metadata else {}
    doc_cols = st.columns(4)
    doc_cols[0].metric("File Name", uploaded.name)
    doc_cols[1].metric("File Size", f"{file_size_kb:.1f} KB")
    doc_cols[2].metric(
        "Format",
        f"PDF ({meta_dict.get('page_count', 1)} pages)" if is_pdf else str(meta_dict.get("format", "Image")),
    )
    doc_cols[3].metric(
        "Editing Software",
        "Detected" if forensic.metadata and forensic.metadata.features.software_editing_indicator else ("Present" if meta_dict.get("software") else "None"),
    )

    with st.expander("👁️ Document Visual Preview & Extracted Text", expanded=False):
        if preview_image is not None:
            st.image(
                preview_image,
                caption=f"Preview: {uploaded.name} (Page {page_index + 1})" if is_pdf else f"Preview: {uploaded.name}",
                use_container_width=True,
            )
        if extracted_text.strip():
            st.markdown("**Text Content Extracted from Document:**")
            st.text_area("Document Content", extracted_text, height=150, disabled=True, key="new_claim_text_area")

    # Forensic Details
    st.markdown("#### 🔬 Detailed Forensics Inspection")
    col1, col2, col3 = st.columns(3)
    if forensic.ela is not None:
        col1.markdown("**1. Error Level Analysis (ELA)**")
        col1.image(forensic.ela.ela_image, use_container_width=True)
        col1.metric("High-error pixel ratio", f"{forensic.ela.features.high_error_pixel_ratio:.3f}")
        col1.caption("Measures recompression error across pixel blocks to highlight spliced/pasted elements.")
    else:
        col1.info("ELA unavailable for this file.")

    if forensic.copy_move is not None:
        col2.markdown("**2. Copy-Move Forgery Detection**")
        if forensic.copy_move.visualization is not None:
            col2.image(forensic.copy_move.visualization, use_container_width=True)
        col2.metric(
            "Duplicated Regions",
            "Detected" if forensic.copy_move.potential_duplicate_regions_found else "None Found",
        )
        col2.caption("Uses ORB keypoint matching to detect duplicated signatures, hospital stamps, or cloned amounts.")
    else:
        col2.info("Copy-move detection unavailable for this file.")

    if forensic.metadata is not None:
        col3.markdown("**3. Metadata & Revision Analysis**")
        if is_pdf:
            if meta_dict.get("author"):
                col3.write(f"• **Author:** {meta_dict.get('author')}")
            if meta_dict.get("software"):
                col3.write(f"• **Software Tag:** `{meta_dict.get('software')}`")
            if meta_dict.get("datetime_original"):
                col3.write(f"• **Created:** {meta_dict.get('datetime_original')}")
            if meta_dict.get("datetime_modified"):
                col3.write(f"• **Modified:** {meta_dict.get('datetime_modified')}")
        else:
            col3.write(f"• **Dimensions:** {meta_dict.get('width', 0)} × {meta_dict.get('height', 0)} px")
            if meta_dict.get("camera_make") or meta_dict.get("camera_model"):
                cam = " ".join(filter(None, [meta_dict.get("camera_make"), meta_dict.get("camera_model")]))
                col3.write(f"• **Camera:** {cam}")
            if meta_dict.get("software"):
                col3.write(f"• **Software Tag:** `{meta_dict.get('software')}`")
            if meta_dict.get("datetime_original"):
                col3.write(f"• **Created:** {meta_dict.get('datetime_original')}")
            if meta_dict.get("datetime_modified"):
                col3.write(f"• **Modified:** {meta_dict.get('datetime_modified')}")

        st_indicators = forensic.metadata.analysis.get("indicators", [])
        if st_indicators:
            col3.markdown("**Metadata Indicators:**")
            for indicator in st_indicators:
                col3.write(f"- {indicator}")
    else:
        col3.info("Metadata analysis unavailable for this file.")

    # Fraud Risk Prediction
    st.markdown("---")
    st.subheader(f"🎯 Fraud Risk Assessment for New Claim `{new_claim_id}`")
    st.caption("Combines the uploaded document's forensic findings with domain risk models and SHAP explainability:")
    _render_assessment(assessment)


def _render_historical_claims_tab(domain: str, model, dataset, model_dir) -> None:
    claim_keys = list(dataset.X[CLAIM_KEY_COL])
    if not claim_keys:
        st.warning("No claims loaded for this sample size -- try increasing it in the sidebar.")
        return

    claim_key = st.selectbox("Select a claim to review", claim_keys)
    assert claim_key is not None

    _render_claim_info(domain, dataset, claim_key)
    _render_graph_panel(dataset, claim_key)

    row = claim_row(dataset, claim_key)
    assessment = assess_claim(domain, model, row, DEFAULT_RISK_THRESHOLDS)
    st.subheader("Fraud risk assessment")
    _render_assessment(assessment)

    _render_forensics_panel(domain, model, dataset, claim_key)
    _render_eval_metrics(model_dir)


def main() -> None:
    _render_header()

    domain, sampling_params = _sidebar_domain_and_sampling()
    model, model_dir = _load_model(domain)
    dataset = _load_dataset(domain, sampling_params)

    tab_new, tab_hist = st.tabs(["📄 Upload & Analyze New Claim", "📊 Historical Claims Database"])

    with tab_new:
        _render_new_claim_tab(domain, model)

    with tab_hist:
        _render_historical_claims_tab(domain, model, dataset, model_dir)


main()
